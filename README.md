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


- **Black Level Correction:** Adjusts pixel values to accurately represent the darkest areas in an image, eliminating any offset caused by the camera sensor. For each (R, Gr, Gb, B) data point, the offset is corrected using the formula:
  \[
  \text{corrected\_val} = \frac{{\text{pixel\_val} - \text{black\_level}}}{{\text{white\_level} - \text{black\_level}}} \times (2^{\text{bit\_depth}} - 1)
  \]
  This normalization ensures that the pixel values are accurately scaled between the black and white levels of the sensor.

- **Bad Pixel Correction:** Utilizes a median filter to correct pixel values. Pixels that significantly deviate from the median value of their neighboring pixels are identified as bad pixels and are replaced with the median value of their neighborhood. This helps in reducing noise and artifacts caused by defective pixels in the sensor.

- **Channel Gain White Balance:** Applies gain to each channel to balance the color. The transformation is defined as:
  \[
  \begin{bmatrix}
  R_{\text{out}} \\
  G_{\text{out}} \\
  B_{\text{out}}
  \end{bmatrix}
  =
  \begin{bmatrix}
  r_g & 0 & 0 \\
  0 & g_g & 0 \\
  0 & 0 & b_g
  \end{bmatrix}
  \begin{bmatrix}
  R_{\text{in}} \\
  G_{\text{in}} \\
  B_{\text{in}}
  \end{bmatrix}
  \]
  The gain for each channel \([r_g, g_g, b_g]\) can be computed by the maximum brightness point method, which finds the values corresponding to white color. The gains are determined as:
  \[
  [r_g, g_g, b_g] = \left[\frac{R_{\text{white}}}{R_{\text{max}}}, \frac{G_{\text{white}}}{G_{\text{max}}}, \frac{B_{\text{white}}}{B_{\text{max}}}\right]
  \]
  This ensures that the white balance is correctly adjusted, making the colors in the image appear natural.

- **Demosaic:** Converts the Bayer pattern image into a full-color image. The Bayer pattern captures images with a single color channel per pixel (R, G, or B), and the demosaicing process interpolates the missing color channels for each pixel to produce a complete RGB image. This step is crucial for converting raw sensor data into a usable image with accurate colors.

These features are essential components of an Image Signal Processing (ISP) pipeline, which collectively enhance the quality and accuracy of the captured images, making them suitable for further processing and analysis.




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

