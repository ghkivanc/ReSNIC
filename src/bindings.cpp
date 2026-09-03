#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "snic.hpp"
#include <vector>

namespace py = pybind11;

void runSNIC_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> image,
    int width,
    int height,
    int channels,
    int numSuperpixels,
    double compactness,
    bool doRGBtoLAB,
    py::array_t<int> labels,
    py::array_t<int> numLabels
) {
    // Flattened buffer pointer
    auto buf = image.request();
    double* img_ptr = static_cast<double*>(buf.ptr);

    // Output arrays
    auto lbl_buf = labels.request();
    auto nl_buf = numLabels.request();
    int* labels_ptr = static_cast<int*>(lbl_buf.ptr);
    int* numLabels_ptr = static_cast<int*>(nl_buf.ptr);

    // Call the C API
    SNIC_main(
        img_ptr,       // flat pointer
        width,
        height,
        channels,
        numSuperpixels,
        compactness,
        doRGBtoLAB,
        labels_ptr,
        numLabels_ptr
    );
}

PYBIND11_MODULE(snic_ext, m) {
    m.def("SNIC_main", &runSNIC_wrapper,
          "SNIC segmentation (double** image)");
}

