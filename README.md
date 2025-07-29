#  Autodiff Phase Retrieval for Computer-Generated Holography

This project implements a **phase retrieval algorithm** using **automatic differentiation** and **gradient descent** in PyTorch. The goal is to generate a phase pattern for a **Spatial Light Modulator (SLM)** that reconstructs a target intensity image through Fourier optics for **computer generated holography**.

---

##  Overview

Phase retrieval is a fundamental problem in computational holography. Given a target image (intensity), the algorithm estimates the phase pattern that, when combined with a known amplitude and passed through an optical system (Fourier Transform), reconstructs the desired image.

This project:
- Loads a grayscale target image.
- Initializes a random phase and fixed amplitude.
- Optimizes the phase using **PyTorch autograd** and the **ADAM optimizer**.
- Saves the recovered phase pattern to an `.h5` file.

---

##  How It Works

### 1. Target Image
The user provides a grayscale image (e.g., `sakura.jpg`) that serves as the target intensity in the reconstruction plane. The image is resized to match the SLM dimensions (1280×1024).

### 2. Initialization
A random phase is generated, and the amplitude is set uniformly.

### 3. Optimization
The forward model uses a Fourier transform to simulate the optical propagation:
- Complex field at the SLM: u = A · exp(i·φ)
- Image plane: U = |F(u)|

The loss function minimizes the difference between the simulated amplitude and the target:
```python
loss = torch.sqrt(torch.sum((U - targetA)**2))
```

### Example Outputs

| Target Intensity | Recovered Phase | Reconstructed Intensity | Error Curve |
|------------------|------------------|--------------------------|-------------|
| ![target](target_samples/sakura.jpg) | ![phase](https://via.placeholder.com/150) | ![reconstructed](https://via.placeholder.com/150) | ![error](https://via.placeholder.com/150) |


---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org)
- NumPy
- Matplotlib
- Pillow (PIL)
- h5py

Install dependencies with:

```bash
pip install torch numpy matplotlib pillow h5py
```
