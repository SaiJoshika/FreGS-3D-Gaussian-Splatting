# Mip-Splatting – Alias-free 3D Gaussian Splatting:

"Mip-Splatting" extends 3D Gaussian Splatting (3DGS) to eliminate zoom-dependent artifacts (erosion when zooming in; aliasing/over-bloat when zooming out) by applying principled filtering in both 3D and 2D. It leverages sampling theory (Nyquist limits) to ensure each Gaussian carries only resolvable frequencies and is rendered with a dynamic, zoom-aware blur.

## Problems Addressed:
1. Zoom-In “Erosion”:
Fine details shrink below a pixel and vanish, because the fixed screen dilation is too small.
2. Zoom-Out “Over-Dilation” & Aliasing:
Distant blobs cover fractions of a pixel but still receive full dilation, causing bloated halos and jagged/flickering edges.

## Model
Mip-Splatting builds directly on the 3D Gaussian Splatting representation, so the “model” is:
A mixture of anisotropic 3D Gaussians, each parameterized by
  1. A 3D center,
  2. A 3×3 covariance (shape/orientation), and
  3. view-dependent color encoded as spherical-harmonic coefficients.

## Method 1: 3D Frequency Regularization

### Goal: Remove ultra-high frequencies that no view can resolve.
1. Track each Gaussian’s maximal sampling rate (smallest on-screen footprint) during training.
2. Compute a 3D low-pass Gaussian whose standard deviation = Nyquist limit of that footprint.
3. Analytically combined with the original covariance.

## Method 2: 2D Mipmap Filter (Zoom-Aware Blur)
### Core idea: Replace the one-size-fits-all screen blur with a blur whose size automatically matches how many pixels each projected blob covers.
How it works:
1. Treat each 2D splat like a tiny texture.
2. Computing its on-screen footprint in pixels.
3. Choose a Gaussian whose standard deviation equals half that footprint (mimicking a box filter over the pixel area).
4. Blur with that Gaussian at render time.

## Implementation:

Clone the repository and create an anaconda environment using

git clone

cd mip-splatting

conda create -y -n mip-splatting python=3.8

conda activate mip-splatting

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f
https://download.pytorch.org/whl/torch_stable.html

![image](https://github.com/user-attachments/assets/55d3d864-2644-4240-ad7f-e4adc6ec0df0)


conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

![image](https://github.com/user-attachments/assets/b11fa155-179d-4f62-8af2-8c1252fdde92)

pip install submodules/diff-gaussian-rasterization

pip install submodules/simple-knn/

![image](https://github.com/user-attachments/assets/4e33ef45-2bb7-4a6e-a6b5-c97a67cff556)


### Blender Dataset

Please download and unzip nerf_synthetic.zip from the [NeRF's official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). 
Then generate multi-scale blender dataset with

python convert_blender_data.py --blender_dir nerf_synthetic/ --out_dir multi-scale


### Mip-NeRF 360 Dataset

Please download the data from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and request the authors for the treehill and flowers scenes.

#single-scale training and multi-scale testing on NeRF-synthetic dataset

python scripts/run_nerf_synthetic_stmt.py

#multi-scale training and multi-scale testing on NeRF-synthetic dataset

python scripts/run_nerf_synthetic_mtmt.py

#single-scale training and single-scale testing on the mip-nerf 360 dataset

python scripts/run_mipnerf360.py

#single-scale training and multi-scale testing on the mip-nerf 360 dataset

python scripts/run_mipnerf360_stmt.py 

![image](https://github.com/user-attachments/assets/aa93ef1b-bebe-4105-a5cb-a34fa3922ab6)

## Input
A folder of RGB photographs of your scene.
Each image must have known intrinsics (focal length, principal point) and extrinsics (camera pose), typically produced by a Structure-from-Motion tool like COLMAP.

## Expected Results
1. Close-up & wide-angle renders free of erosion or bloating.
2. Smooth transitions across zoom levels without retraining.
3. Real-time performance maintained with minimal overhead.

<img width="398" alt="image" src="https://github.com/user-attachments/assets/5783b749-a666-4bc8-a81b-9f4c44148134" />

## Limitations
### 1.Approximation Error in 2D Mip Filter

Mip-Splatting uses a Gaussian to approximate the ideal box filter of a camera pixel for efficiency. When a projected Gaussian splat is very small on screen (e.g. under extreme zoom-out), this approximation deviates more from the true box filter, introducing small rendering errors .

### 2.Extra Training Overhead

To enforce Nyquist limits, the sampling rate of each 3D Gaussian must be recomputed every m = 100 training iterations. Currently this is done in PyTorch, which incurs a modest slowdown. A dedicated CUDA implementation or a precomputed data structure (since sampling rates depend only on fixed camera poses/intrinsics) could reduce this overhead.

### 3.Residual Frequency Leakage at Extremes

Although the 3D smoothing and 2D Mip filters together eliminate most zoom artifacts, very extreme changes in sampling rate (far beyond those in the training set) may still reveal slight aliasing or smoothing errors due to the Gaussian approximations.




