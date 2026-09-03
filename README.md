# ReSNIC: Rapid and Efficient Superpixel Segmentation with NVIDIA CUDA

A high-performance GPU-accelerated implementation of the SNIC (Superpixel Nearest Neighbor Clustering) superpixel segmentation algorithm using NVIDIA CUDA and C++.

## What is ReSNIC?

ReSNIC provides a fast, CUDA-accelerated implementation of superpixel segmentation using the SNIC algorithm. Superpixels are perceptually meaningful atomic regions in images that respect object boundaries. This implementation dramatically accelerates superpixel generation by leveraging GPU parallel processing, making it suitable for real-time and batch image processing applications.

### Key Features

- **GPU-Accelerated**: Implemented in CUDA for maximum performance on NVIDIA GPUs
- **Python Integration**: Easy-to-use Python interface via pybind11
- **Flexible Input**: Supports RGB and arbitrary multi-channel images
- **LAB Color Space**: Automatic RGB→LAB color conversion for perceptually uniform segmentation
- **Tunable Compactness**: Control the balance between spatial and color coherence in superpixels
- **Hybrid Implementation**: CPU preprocessing with GPU-accelerated core algorithm

## Installation

### Prerequisites

- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0 or higher
- **CUDA Toolkit**: CUDA 11.0 or later
- **CMake**: Version 3.18 or higher
- **Python**: Python 3.6 or later with development headers
- **Compiler**: C++17 compatible compiler (GCC, Clang, or MSVC)

### Build from Source

1. **Clone the repository**
   ```bash
   git clone https://github.com/ghkivanc/ReSNIC.git
   cd ReSNIC
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure CUDA architecture in CMakeLists.txt**
   
   Edit `CMakeLists.txt` and set your GPU's compute capability:
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES YOUR_ARCH)  # e.g., 75 for RTX 2080, 86 for RTX 3080
   ```
   
   Common architectures:
   - `60` - P100, Quadro P5000
   - `70` - V100, Quadro V100
   - `75` - RTX 2070/2080, T4, Quadro RTX
   - `80` - A100, A10
   - `86` - RTX 3060/3070/3080/3090, RTX 4000
   - `89` - RTX 4080/4090

4. **Build the project**
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

5. **Install Python package**
   ```bash
   cd build
   pip install -e .
   ```

## Usage

### Python Example

```python
import numpy as np
from PIL import Image
import snic_ext

# Load image
image = Image.open('image.jpg')
image_array = np.array(image, dtype=np.float64)  # Shape: (H, W, 3)

# Prepare output arrays
height, width, channels = image_array.shape
num_superpixels = 400  # Desired number of superpixels
labels = np.zeros((height * width,), dtype=np.int32)
num_labels = np.zeros(1, dtype=np.int32)

# Run SNIC segmentation
# Arguments: image, width, height, channels, num_superpixels, compactness, doRGBtoLAB
snic_ext.SNIC_main(
    image_array.flatten(),
    width,
    height,
    channels,
    num_superpixels,
    compactness=0.1,  # Balance spatial vs. color coherence
    doRGBtoLAB=True   # Convert RGB to LAB color space
)

# Reshape output
labels = labels.reshape((height, width))

# Visualize superpixels (optional)
import matplotlib.pyplot as plt
from skimage import mark_boundaries

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(image_array / 255.0)
ax1.set_title('Original Image')
ax2.imshow(mark_boundaries(image_array / 255.0, labels))
ax2.set_title(f'Superpixels (k={num_labels[0]})')
plt.show()
```

### Parameters

- **image** (ndarray): Flattened image array (C-contiguous, float64)
- **width** (int): Image width in pixels
- **height** (int): Image height in pixels
- **channels** (int): Number of color channels (typically 3 for RGB)
- **num_superpixels** (int): Target number of superpixels
- **compactness** (float): Spatial compactness parameter (0.0-1.0)
  - Lower values → color-based segmentation
  - Higher values → spatial regularity
- **doRGBtoLAB** (bool): Whether to convert RGB to LAB color space (recommended: True)

## Citation

If you use ReSNIC in your research, please cite:

```bibtex
@software{ReSNIC2026,
  author = {Kivanc, G.},
  title = {ReSNIC: Rapid and Efficient Superpixel Segmentation with NVIDIA CUDA},
  year = {2026},
  url = {https://github.com/ghkivanc/ReSNIC}
}
```

### Original SNIC Algorithm

If you use the SNIC algorithm itself, please also cite the original paper:

```bibtex
@inproceedings{Achanta2016SNIC,
  author = {Achanta, Radhakrishna and Susstrunk, Sabine},
  title = {Superpixels and Polygons using Simple Linear Iterative Clustering},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2016},
  pages = {4651--4660},
  doi = {10.1109/CVPR.2016.502}
}
```

## License

This project retains the original EPFL license from the SNIC algorithm. See the source files for full license details.

## References

- Achanta, R., & Susstrunk, S. (2016). Superpixels and Polygons using Simple Linear Iterative Clustering. IEEE Conference on Computer Vision and Pattern Recognition.
- NVIDIA CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/

## Troubleshooting

- **CUDA out of memory**: Reduce `num_superpixels` or process smaller images
- **Compilation errors**: Verify CUDA_ARCHITECTURES matches your GPU and CUDA toolkit version
- **Poor segmentation results**: Adjust the `compactness` parameter or number of superpixels
