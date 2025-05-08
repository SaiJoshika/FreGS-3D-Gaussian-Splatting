# Mip-Splatting: Alias-Free 3D Gaussian Splatting

A principled extension to 3D Gaussian Splatting that eliminates zoom-dependent artifacts by enforcing frequency constraints in 3D and applying a dynamic, zoom-aware filter in screen space.

## 1. Introduction

<img width="504" alt="image" src="https://github.com/user-attachments/assets/a8439836-0c98-475b-8fd8-0f7ed296a13e" />


Mip-Splatting addresses the limitations of standard 3D Gaussian Splatting (3DGS) when changing the camera’s sampling rate (via focal length or distance). While 3DGS excels at real-time novel view synthesis, it suffers from:

* **Zoom-In Erosion:** Thin structures vanish when magnified because a fixed screen-space dilation is too small.
* **Zoom-Out Aliasing & Over-Dilation:** Distant blobs appear bloated and produce jagged/flickering edges due to constant dilation and lack of low-pass filtering.

This work introduces two lightweight, principled filters—applied during training and rendering—to guarantee artifact-free results across arbitrary zoom levels.


## 2. Methodology

### 2.1 3D Frequency Regularization

1. **Compute Maximal Sampling Rate:** Track each Gaussian’s smallest pixel footprint across all training views.
2. **Nyquist Covariance:** Derive a low-pass Gaussian whose standard deviation equals half that footprint (Nyquist limit).
3. **Analytic Fusion:** Add the Nyquist covariance to the original 3D covariance per Gaussian:
   $\Sigma_{3D}' = \Sigma_{3D} + \Sigma_{nyquist}$

### 2.2 2D Mip Filter

1. **Screen-Space Footprint:** For each projected splat, compute its on-screen width *w* via the projection Jacobian.
2. **Adaptive Dilation:** Set 2D blur σ₂D = *w*/2 and incorporate into the 2D covariance

3. **Result:** Each splat uses just enough blur—more when small (avoid aliasing), less when large (preserve detail).

## 3. Prerequisites and Installation
# Installation
Clone the repository and create an anaconda environment using
```
git clone git@github.com:autonomousvision/mip-splatting.git
cd mip-splatting

conda create -y -n mip-splatting python=3.8
conda activate mip-splatting 

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
```

# Dataset
## Blender Dataset
Please download and unzip nerf_synthetic.zip from the [NeRF's official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Then generate multi-scale blender dataset with
```
python convert_blender_data.py --blender_dir nerf_synthetic/ --out_dir multi-scale
```

## Mip-NeRF 360 Dataset
Please download the data from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and request the authors for the treehill and flowers scenes.

# Training and Evaluation
```
# single-scale training and multi-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_stmt.py 

# multi-scale training and multi-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_mtmt.py 

# single-scale training and single-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360.py 

# single-scale training and multi-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360_stmt.py 
```

# Online viewer
After training, you can fuse the 3D smoothing filter to the Gaussian parameters with
```
python create_fused_ply.py -m {model_dir}/{scene} --output_ply fused/{scene}_fused.ply"
```
Then use our [online viewer](https://niujinshuchong.github.io/mip-splatting-demo) to visualize the trained model.


## 4. Experimental Results

* **Single-Scale Training, Multi-Scale Testing:** Mip-Splatting maintains or improves PSNR/SSIM/LPIPS across focal scales (½×, 2×, 4×) compared to 3DGS and EWA variants.
* **Benchmarks:** On Blender \[28] and Mip-NeRF 360 \[2], our method outperforms prior art in out-of-distribution zoom settings while matching state-of-the-art at training scale.
* **Qualitative:** Eliminates erosion of thin structures and prevents bloated halos or shimmering at distant views.

 <img width="248" alt="image" src="https://github.com/user-attachments/assets/63cd01a5-15e0-45f0-80fc-c3c055057e26" />
