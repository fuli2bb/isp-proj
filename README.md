## Overview

This is a study project focused on developing algorithms for a camera imaging pipeline. 
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

- **Image Enhancement:** Techniques for improving the visual quality of images.
- **Noise Reduction:** Methods to remove or reduce noise from images.
- **Edge Detection:** Algorithms to identify and highlight edges in images.
- **Image Filtering:** Application of various filters to images for different effects.
- **Transformation:** Geometric transformations such as rotation, scaling, and translation.
- **Detailed Console Output:** Comprehensive logs and process tracking for debugging and analysis.

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

### Run command

```bash
./build/isp config.ini
