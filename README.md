# ReSNIC: SNIC Superpixel Segmentation with CUDA

A high-performance GPU-accelerated implementation of the SNIC (Simple Non-Iterative Clustering) superpixel segmentation algorithm using CUDA and C++.

## What is ReSNIC?

ReSNIC provides a fast, CUDA-accelerated implementation of superpixel segmentation using the SNIC algorithm. Superpixels are perceptually meaningful atomic regions in images that respect object boundaries. This implementation dramatically accelerates superpixel generation by leveraging GPU parallel processing, making it suitable for real-time and batch image processing applications.

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
   I personally suggest using conda/miniconda to install all required packages in isolation.
   ```bash
   conda create -n my_env --file requirements.txt
   ```
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure CUDA architecture in CMakeLists.txt**
   
   Edit `CMakeLists.txt` and set your GPU's compute capability:
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES YOUR_ARCH)  # e.g., 75 for RTX 2080, 86 for RTX 3080
   ```
   
5. **Build & install the project**
   Before you build I suggest you check the snic_kernels.cu as it includes some hyper-parameters that are decided in compile time.
   ```bash
   ./scripts/compile.sh
   ```

## Usage

### Python Example
An example use can be found in tests/test.py

## Citation

If you use ReSNIC in your research, please cite:

```bibtex
@inproceedings{tas2026dataparallel,
  author={Ta{\c{s}}, K{\i}van{\c{c}} and Akg{\"u}n, Toygar},
  title={Data-Parallel CUDA Implementation of the SNIC Super-Pixel Algorithm},
  booktitle={2026 IEEE International Conference on Image Processing (ICIP)},
  year={2026},
  month={Sep},
  doi={10.1109/icip61757.2026.11630136},
  organization={IEEE}
}
```

### Original SNIC Algorithm

If you use the SNIC algorithm itself, please also cite the original paper:

```bibtex
@inproceedings{snic_cvpr17, author = {Achanta, Radhakrishna and Susstrunk, Sabine},
title = {Superpixels and Polygons using Simple Non-Iterative Clustering},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, year = {2017} }
}
```

## References

- Achanta, R., & Susstrunk, S. (2016). Superpixels and Polygons using Simple Linear Iterative Clustering. IEEE Conference on Computer Vision and Pattern Recognition.
- NVIDIA CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/

## Troubleshooting

- **Heap size exceeded**: Likely due to either your arch or the heap_size parameter inside snic_kernels.cu. Refer to the file for further information
- **Poor segmentation results**: Adjust the `compactness` parameter or number of superpixels
