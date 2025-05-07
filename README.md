# Mip-Splatting: Alias-Free 3D Gaussian Splatting

A principled extension to 3D Gaussian Splatting that eliminates zoom-dependent artifacts by enforcing frequency constraints in 3D and applying a dynamic, zoom-aware filter in screen space.
![Screenshot 2025-05-06 212405](https://github.com/user-attachments/assets/6af221d8-1fde-406a-9d89-ce2eb4156243)



## 1. Introduction

Mip-Splatting addresses the limitations of standard 3D Gaussian Splatting (3DGS) when changing the camera’s sampling rate (via focal length or distance). While 3DGS excels at real-time novel view synthesis, it suffers from:

* **Zoom-In Erosion:** Thin structures vanish when magnified because a fixed screen-space dilation is too small.
* **Zoom-Out Aliasing & Over-Dilation:** Distant blobs appear bloated and produce jagged/flickering edges due to constant dilation and lack of low-pass filtering.

This work introduces two lightweight, principled filters—applied during training and rendering—to guarantee artifact-free results across arbitrary zoom levels.


## 2. Key Contributions

1. **3D Frequency Regularization**

   * A 3D low-pass filter that constrains each Gaussian’s maximal frequency based on its highest sampling rate during training.
2. **2D Mip Filter**

   * A screen-space, zoom-aware Gaussian approximation of a box filter that adapts blur to each splat’s pixel footprint, eliminating aliasing and over-dilation.


## 3. Methodology

### 3.1 3D Frequency Regularization

1. **Compute Maximal Sampling Rate:** Track each Gaussian’s smallest pixel footprint across all training views.
2. **Nyquist Covariance:** Derive a low-pass Gaussian whose standard deviation equals half that footprint (Nyquist limit).
3. **Analytic Fusion:** Add the Nyquist covariance to the original 3D covariance per Gaussian:
   $\Sigma_{3D}' = \Sigma_{3D} + \Sigma_{nyquist}$

### 3.2 2D Mip Filter

1. **Screen-Space Footprint:** For each projected splat, compute its on-screen width *w* via the projection Jacobian.
2. **Adaptive Dilation:** Set 2D blur σ₂D = *w*/2 and incorporate into the 2D covariance

3. **Result:** Each splat uses just enough blur—more when small (avoid aliasing), less when large (preserve detail).


## 4. Experimental Results

* **Single-Scale Training, Multi-Scale Testing:** Mip-Splatting maintains or improves PSNR/SSIM/LPIPS across focal scales (½×, 2×, 4×) compared to 3DGS and EWA variants.
* **Benchmarks:** On Blender \[28] and Mip-NeRF 360 \[2], our method outperforms prior art in out-of-distribution zoom settings while matching state-of-the-art at training scale.
* **Qualitative:** Eliminates erosion of thin structures and prevents bloated halos or shimmering at distant views.

---

## 5. Limitations & Future Work

* **Gaussian Approximation Error:** The 2D Gaussian approximation of a box filter can introduce small errors at extreme zoom levels.
* **Training Overhead:** Computing per-Gaussian sampling rates every *m* iterations adds minimal overhead; optimized CUDA kernels could further reduce this cost.
* **Extension:** Exploring more accurate filter shapes or precomputed data structures for real-time sampling-rate lookups.
