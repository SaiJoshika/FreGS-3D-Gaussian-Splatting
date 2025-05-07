Mip-Splatting – Alias-free 3D Gaussian Splatting:
Mip-Splatting is a method for alias-free novel view synthesis using 3D Gaussian Splatting. It addresses key artifacts (like erosion, dilation, and aliasing) seen in previous 3DGS methods when rendering at sampling rates different from training.

3D Gaussian Splatting (3DGS):
1) Represents scenes with 3D Gaussians.
2) Renders by projecting them to 2D via splatting.
3) Uses a 2D dilation filter in screen space.
4) Suffers from rendering artifacts at different zoom levels due to lack of frequency control.

Key Concepts:
3D Smoothing Filter:
1. Applies a low-pass filter in 3D space.
2. Based on the Nyquist-Shannon theorem.
3. Constrains the maximum frequency of each Gaussian.
4. Prevents high-frequency artifacts when zooming in.

2D Mip Filter:
1. Replaces the 2D dilation.
2. Simulates a 2D box filter (like mipmapping).
3. Approximated with a Gaussian filter covering one pixel.
4. Removes aliasing and improves rendering when zooming out.

Implementation:
1. Recomputes sampling rates every 100 iterations.
2. Lightweight modifications with high performance gain.

Experimental Results:
Evaluation Datasets:
1. Blender Dataset: Used for zoom-out tests.
2. Mip-NeRF 360 Dataset: Used for zoom-in tests.

Performance Summary:
1. Outperforms previous methods like NeRF, 3DGS, EWA splatting across scales.
2. Maintains fidelity in both zoom-in and zoom-out rendering.
3. Especially effective when training at a single scale and testing across many.

Limitations:
1. Approximation of box filter using Gaussian introduces small errors.
2. Slight training overhead due to sampling rate computation.
3. Performance might drop when zooming out extremely far.

Output:

