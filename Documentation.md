Mip-Splatting – Alias-free 3D Gaussian Splatting:

"Mip-Splatting" extends 3D Gaussian Splatting (3DGS) to eliminate zoom-dependent artifacts (erosion when zooming in; aliasing/over-bloat when zooming out) by applying principled filtering in both 3D and 2D. It leverages sampling theory (Nyquist limits) to ensure each Gaussian carries only resolvable frequencies and is rendered with a dynamic, zoom-aware blur.

Problems Addressed:
1. Zoom-In “Erosion”:
• Fine details shrink below a pixel and vanish, because the fixed screen dilation is too small.
2. Zoom-Out “Over-Dilation” & Aliasing:
• Distant blobs cover fractions of a pixel but still receive full dilation, causing bloated halos and jagged/flickering edges.

Method 1: 3D Frequency Regularization
Goal: Remove ultra-high frequencies that no view can resolve.
• Track each Gaussian’s maximal sampling rate (smallest on-screen footprint) during training.
• Compute a 3D low-pass Gaussian whose standard deviation = Nyquist limit of that footprint.
• Analytically combined with the original covariance.

Method 2: 2D Mipmap Filter (Zoom-Aware Blur)
Core idea: Replace the one-size-fits-all screen blur with a blur whose size automatically matches how many pixels each projected blob covers.
How it works:
1. Treat each 2D splat like a tiny texture.
2. Computing its on-screen footprint in pixels.
3. Choose a Gaussian whose standard deviation equals half that footprint (mimicking a box filter over the pixel area).
4. Blur with that Gaussian at render time.

Implementation:
Clone the repository and create an anaconda environment using
git clone
cd mip-splatting
conda create -y -n mip-splatting python=3.8
conda activate mip-splatting
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f
https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/

Blender Dataset
Please download and unzip nerf_synthetic.zip from the [NeRF's official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). 
Then generate multi-scale blender dataset with
python convert_blender_data.py --blender_dir nerf_synthetic/ --out_dir multi-scale

Mip-NeRF 360 Dataset
Please download the data from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and
request the authors for the treehill and flowers scenes.
# single-scale training and multi-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_stmt.py
# multi-scale training and multi-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_mtmt.py
# single-scale training and single-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360.py
# single-scale training and multi-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360_stmt.py 

Expected Results
• Close-up & wide-angle renders free of erosion or bloating.
• Smooth transitions across zoom levels without retraining.
• Real-time performance maintained with minimal overhead.


![image](https://github.com/user-attachments/assets/62af74e1-8375-4a35-a28e-30f67e5747c6)


