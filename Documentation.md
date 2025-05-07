Mip-Splatting – Alias-free 3D Gaussian Splatting:
Mip-Splatting is a method for alias-free novel view synthesis using 3D Gaussian Splatting. It addresses key artifacts (like erosion, dilation, and aliasing) seen in previous 3DGS methods when rendering at sampling rates different from training.

3D Gaussian Splatting (3DGS):
1) Represents scenes with 3D Gaussians.
2) Renders by projecting them to 2D via splatting.
3) Uses a 2D dilation filter in screen space.
4)Suffers from rendering artifacts at different zoom levels due to lack of frequency control.

Key Concepts:
1. 3D Smoothing Filter:
Applies a low-pass filter in 3D space.
Based on the Nyquist-Shannon theorem.
Constrains the maximum frequency of each Gaussian.
Prevents high-frequency artifacts when zooming in.

2. 2D Mip Filter:
Replaces the 2D dilation.
Simulates a 2D box filter (like mipmapping).
Approximated with a Gaussian filter covering one pixel.
Removes aliasing and improves rendering when zooming out.

Implementation:
Recomputes sampling rates every 100 iterations.
Lightweight modifications with high performance gain.

Experimental Results:
1) Evaluation Datasets
Blender Dataset: Used for zoom-out tests.
Mip-NeRF 360 Dataset: Used for zoom-in tests.
2) Performance Summary
Outperforms previous methods like NeRF, 3DGS, EWA splatting across scales.
Maintains fidelity in both zoom-in and zoom-out rendering.
Especially effective when training at a single scale and testing across many.

Limitations:
Approximation of box filter using Gaussian introduces small errors.
Slight training overhead due to sampling rate computation.
Performance might drop when zooming out extremely far.

Output:

