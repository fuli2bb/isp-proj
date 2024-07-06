## Overview

![simple isp](images/isp_readme.png)


This is a simple project focused on developing algorithms for a camera imaging pipeline. 
The project references [this repository](https://github.com/mushfiqulalam/isp) and utilizes raw image data and the pipeline framework. 
A C++ implementation was developed using OpenCV, and CUDA implementations for certain time-consuming functions were developed. 
For studying purpose, this project avoids directly using OpenCV's built-in features and functions, mainly using `cv::Mat` as a container.

## Implemented Pipeline

- black level correction
- bad pixel correction 
- channel gain while balance
- demosaic
- process local color ratio
- apply color matrix
- gamma
- purple fringe removel
- tone mapping
- equalize histogram
- sharpening


## Features

- **Black Level Correction:** Adjust pixel values to accurately represent the darkest areas in an image, eliminating any offset caused by the camera sensor. For each (R, Gr, Gb, B) data point, correct the offset using the formula:
 \[\frac{{\text{pixel\_val} - \text{black\_level}}}{{\text{white\_level} - \text{black\_level}}} \times (2^{\text{bit\_depth}} - 1)
  \]

- **Bad pixel correction:** Utilizes a median filter to correct pixel values. Pixels that significantly deviate from the median value of their neighboring pixels are replaced with the median value.
- **Channel gain while balance:** Algorithms to identify and highlight edges in images.
- **Demosaic:** Application of various filters to images for different effects.

### Prerequisites

- C++ compiler (e.g., GCC, Clang)
- CMake
- OpenCV library
- CUDA

### Installation

```bash
git clone https://github.com/fuli2bb/isp-proj.git
cd isp-proj
rm -rf build
mkdir build
./build.sh
```
### Run command

```bash
./build/isp config.ini
```

### CUDA functions

At this moment, CUDA implementation of bad_pixel_correction and histogram equalizition functions were provided. Below is a comparsion of runing time between cpu code and  gpu code.

