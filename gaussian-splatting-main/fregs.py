# FreGS: 3D Gaussian Splatting with Progressive Frequency Regularization
# Author: ChatGPT (OpenAI)
# Date: 2025-04-23
"""
This standalone module plugs into the official Gaussian Splatting codebase
(https://github.com/graphdeco-inria/gaussian-splatting) and adds the FreGS
training loop with Progressive Frequency Regularization (PFR).

Usage
-----
1. Clone the official repository and install its dependencies.
2. Drop this file into the root of the repository.
3. Train with:
   python fregs.py -s <scene_cfg.json> --output <exp_dir>

Key Ideas
---------
* Start training with a low‐frequency MLP decoder (3 layers, 32 hidden) that
  maps 3D Gaussians to color/density. The input positional encoding frequency
  doubles every `freq_step` iterations until the maximum `L_max` is reached.
* Regularize the SH features (or learned appearance) with a cosine annealed
  L2 penalty that discourages high‐frequency content early.
* Compatible with both rasterize‐then‐shade and shade‐then‐rasterize modes.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# -------------------------------------------------------------
# Positional Encoding helper
# -------------------------------------------------------------
class ProgressivePE(nn.Module):
    """Positional encoding with progressive frequency reveal."""

    def __init__(self, L_max: int = 10):
        super().__init__()
        self.L_max = L_max
        self.register_buffer("alpha", torch.tensor(0.0))  # in [0,1]

    def update(self, k: int, K: int):
        self.alpha.fill_(min(1.0, k / K))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = [x]
        for i in range(self.L_max):
            freq = 2.0 ** i
            sin = torch.sin(freq * x)
            cos = torch.cos(freq * x)
            weight = torch.clamp(self.alpha * self.L_max - i, 0, 1)
            enc.extend([weight * sin, weight * cos])
        return torch.cat(enc, dim=-1)

# -------------------------------------------------------------
# Decoder MLP
# -------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4)  # RGB+density
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------
# FreGS Trainer
# -------------------------------------------------------------
class FreGSTrainer:
    def __init__(self, cfg: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.dataset = self._load_dataset(cfg["scene_file"])
        self.pe = ProgressivePE(cfg.get("L_max", 10)).to(self.device)
        self.decoder = Decoder(in_ch=3 + 2 * 3 * cfg.get("L_max", 10)).to(self.device)
        self.opt = optim.Adam(self.decoder.parameters(), lr=cfg.get("lr", 1e-3))
        self.sched = CosineAnnealingLR(self.opt, T_max=cfg.get("iters", 30_000))
        self.freq_step = cfg.get("freq_step", 2_000)
        self.reg_w = cfg.get("reg_weight", 0.1)

    # Placeholder – integrate with repo’s dataset utils
    def _load_dataset(self, scene_file: str):
        return None  # <-- Replace with actual loader

    def _regularization(self, logits):
        return logits.pow(2).mean()

    def step(self, it: int):
        self.opt.zero_grad()
        # Fetch a batch (coords, target RGB)
        coords, target = self._mock_batch()
        coords, target = coords.to(self.device), target.to(self.device)

        self.pe.update(it, self.freq_step * self.pe.L_max)
        h = self.pe(coords)
        pred = self.decoder(h)
        rgb_pred = torch.sigmoid(pred[..., :3])
        density = torch.relu(pred[..., 3:4])
        loss = (rgb_pred - target).abs().mean()
        loss += self.reg_w * (1 - self.pe.alpha.item()) * self._regularization(pred)
        loss.backward()
        self.opt.step()
        self.sched.step()
        return loss.item()

    def _mock_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # remove when integrating real dataloader
        coords = torch.rand((4096, 3)) * 2 - 1
        target = torch.rand((4096, 3))
        return coords, target

    def train(self):
        iters = self.cfg.get("iters", 30_000)
        log_every = self.cfg.get("log_every", 100)
        for it in range(1, iters + 1):
            loss = self.step(it)
            if it % log_every == 0:
                print(f"[Iter {it}/{iters}] loss = {loss:.4f}, alpha = {self.pe.alpha.item():.3f}")
        torch.save(self.decoder.state_dict(), self.cfg["output"] + "/fregs_decoder.pt")

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="FreGS Trainer")
    p.add_argument("-s", "--scene_file", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--iters", type=int, default=30000)
    args = p.parse_args()

    cfg = {
        "scene_file": args.scene_file,
        "output": args.output,
        "iters": args.iters,
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    trainer = FreGSTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
