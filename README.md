# FreGS-3D-Gaussian-Splatting

#### **1. Introduction**  
- **3D Gaussian Splatting (3D-GS)** is a technique for rendering 3D scenes by representing them as a collection of **3D Gaussians**.  
- However, 3D-GS suffers from **over-reconstruction**, leading to blurry images and inefficient Gaussian placement.  
- The paper proposes **FreGS (Frequency Regularized Gaussian Splatting)** to address this issue using **progressive frequency regularization**.  

#### **2. Key Contributions**  
1. **FreGS Framework** – Introduces **frequency regularization** to control Gaussian density and prevent over-reconstruction.  
2. **Frequency Annealing** – Uses a **coarse-to-fine** approach, progressively refining Gaussians by optimizing **low-to-high frequency details**.  
3. **Improved Novel View Synthesis** – FreGS outperforms standard **3D-GS** on datasets like **Mip-NeRF360, Tanks & Temples, and Deep Blending**.  

#### **3. Methodology**  
- **Step 1: Structure-from-Motion (SfM)** → Extracts **3D points** from input images.  
- **Step 2: Gaussian Initialization** → Converts points into **3D Gaussians**.  
- **Step 3: Rendering with Gaussian Splatting** → Projects **3D Gaussians into 2D images**.  
- **Step 4: Frequency Regularization**  
  - Uses **Fourier Transform** to compare the **amplitude and phase components** of rendered images vs. ground truth.  
  - Adjusts Gaussian density using **progressive frequency annealing**.  

#### **4. Experimental Results**  
- **FreGS produces sharper, more detailed images** compared to standard 3D-GS.  
- Evaluations on multiple datasets show **higher reconstruction accuracy**.  

#### **5. Limitations & Future Work**  
- Increased **computational overhead** due to frequency analysis.  
- Struggles with **extremely high-frequency details**.  
- Future work could focus on **real-time performance improvements** and **adaptive Gaussian placement**.  
