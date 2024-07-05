#include "isp_pipeline.hpp"
#include "isp_cuda_func.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        std::exit(-1);
    }
}

__global__ void computeHistogram(const ushort* d_channel, int* d_hist, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        atomicAdd(&d_hist[d_channel[y * cols + x]], 1);
    }
}

__global__ void normalizeCDF(double* d_cdf, double cdfMin, double totalPixels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 65536) {
        d_cdf[tid] = ((d_cdf[tid] - cdfMin) / (totalPixels - cdfMin)) * 65535.0;
    }
}

__global__ void equalizeImage(ushort* d_channel, const double* d_cdf, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int val = d_channel[y * cols + x];
        d_channel[y * cols + x] = static_cast<ushort>(d_cdf[val]);
    }
}

void customEqualizeHist_cuda(cv::Mat& channel) {
    if (channel.type() != CV_16U) {
        std::cerr << "Error: The input channel must be of type CV_16U" << std::endl;
        return;
    }

    ushort* d_channel;
    int rows = channel.rows;
    int cols = channel.cols;
    size_t numPixels = rows * cols;

    checkCuda(cudaMalloc(&d_channel, numPixels * sizeof(ushort)), "Failed to allocate device memory for d_channel");
    checkCuda(cudaMemcpy(d_channel, channel.ptr<ushort>(), numPixels * sizeof(ushort), cudaMemcpyHostToDevice), "Failed to copy channel data to device");

    int histSize = 65536;
    int* d_hist;
    checkCuda(cudaMalloc(&d_hist, histSize * sizeof(int)), "Failed to allocate device memory for d_hist");
    checkCuda(cudaMemset(d_hist, 0, histSize * sizeof(int)), "Failed to set d_hist to zero");

    dim3 blockSize(32, 32);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    computeHistogram<<<gridSize, blockSize>>>(d_channel, d_hist, rows, cols);
    checkCuda(cudaDeviceSynchronize(), "Failed to synchronize after computeHistogram");

    int* h_hist = new int[histSize];
    checkCuda(cudaMemcpy(h_hist, d_hist, histSize * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy histogram data to host");

    double* h_cdf = new double[histSize];
    h_cdf[0] = h_hist[0];
    for (int i = 1; i < histSize; ++i) {
        h_cdf[i] = h_cdf[i - 1] + h_hist[i];
    }

    double cdfMin = h_cdf[0];
    double totalPixels = h_cdf[histSize - 1];

    double* d_cdf;
    checkCuda(cudaMalloc(&d_cdf, histSize * sizeof(double)), "Failed to allocate device memory for d_cdf");
    checkCuda(cudaMemcpy(d_cdf, h_cdf, histSize * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy CDF data to device");

    int threadsPerBlock = 256;
    int blocksPerGrid = (histSize + threadsPerBlock - 1) / threadsPerBlock;
    normalizeCDF<<<blocksPerGrid, threadsPerBlock>>>(d_cdf, cdfMin, totalPixels);
    checkCuda(cudaDeviceSynchronize(), "Failed to synchronize after normalizeCDF");

    equalizeImage<<<gridSize, blockSize>>>(d_channel, d_cdf, rows, cols);
    checkCuda(cudaDeviceSynchronize(), "Failed to synchronize after equalizeImage");

    checkCuda(cudaMemcpy(channel.ptr<ushort>(), d_channel, numPixels * sizeof(ushort), cudaMemcpyDeviceToHost), "Failed to copy equalized image data to host");

    cudaFree(d_channel);
    cudaFree(d_hist);
    cudaFree(d_cdf);

    delete[] h_hist;
    delete[] h_cdf;
}

void isp_pipeline::equalizeHistColor_cuda() {
    // Ensure the image type is CV_32F
    if (cv_mat_data.type() != CV_32FC3) {
        std::cerr << "Error: The input image must be of type CV_32F" << std::endl;
        return;
    }
    // Convert CV_32F to CV_16U for processing
    cv_mat_data.convertTo(cv_mat_data, CV_16UC3);

    // Split the image into B, G, R channels
    std::vector<cv::Mat> channels(3);
    cv::split(cv_mat_data, channels);

    // Equalize each channel independently
    for (int i = 0; i < 3; ++i) {
        customEqualizeHist_cuda(channels[i]);
    }
    // Merge the channels back
    cv::merge(channels, cv_mat_data);

    // Convert back to CV_32F
    cv_mat_data.convertTo(cv_mat_data, CV_32FC3);
}






__global__ void badPixelCorrectionKernel(float* img, int width, int height, int neighborhood_size, float threshold, int no_of_pixel_pad) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int padded_width = width + 2 * no_of_pixel_pad;
    int padded_height = height + 2 * no_of_pixel_pad;

    float mid_pixel_val = img[(y + no_of_pixel_pad) * padded_width + (x + no_of_pixel_pad)];

    // Extract the neighborhood into an array
    float neighborhood[9]; // assuming neighborhood_size is 7, change size if necessary
    int k = 0;
    for (int ni = -no_of_pixel_pad; ni <= no_of_pixel_pad; ++ni) {
        for (int nj = -no_of_pixel_pad; nj <= no_of_pixel_pad; ++nj) {
            neighborhood[k++] = img[(y + no_of_pixel_pad + ni) * padded_width + (x + no_of_pixel_pad + nj)];
        }
    }

    // Calculate the median of the neighborhood
    for (int i = 0; i < k - 1; ++i) {
        for (int j = 0; j < k - i - 1; ++j) {
            if (neighborhood[j] > neighborhood[j + 1]) {
                float temp = neighborhood[j];
                neighborhood[j] = neighborhood[j + 1];
                neighborhood[j + 1] = temp;
            }
        }
    }
    float median_val = neighborhood[k / 2];

    // If the middle pixel value deviates significantly from the median value,
    // consider it as a bad pixel and replace it with the median value
    if (fabs(mid_pixel_val - median_val) > threshold) {
        img[(y + no_of_pixel_pad) * padded_width + (x + no_of_pixel_pad)] = median_val;
    }
}

void isp_pipeline::bad_pixel_correction_cuda(const int neighborhood_size) {
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Running bad pixel correction..." << std::endl;

    if ((neighborhood_size % 2) == 0) {
        std::cerr << "neighborhood_size should be an odd number" << std::endl;
        return;
    }

    // Separate the image into four quarters for Bayer pattern
    std::vector<cv::Mat> quarters(4);
    quarters[0] = cv::Mat(cv_mat_data.rows / 2, cv_mat_data.cols / 2, CV_32F);
    quarters[1] = cv::Mat(cv_mat_data.rows / 2, cv_mat_data.cols / 2, CV_32F);
    quarters[2] = cv::Mat(cv_mat_data.rows / 2, cv_mat_data.cols / 2, CV_32F);
    quarters[3] = cv::Mat(cv_mat_data.rows / 2, cv_mat_data.cols / 2, CV_32F);

    for (int i = 0; i < cv_mat_data.rows; ++i) {
        for (int j = 0; j < cv_mat_data.cols; ++j) {
            int idx = (i % 2) * 2 + (j % 2);
            quarters[idx].at<float>(i / 2, j / 2) = cv_mat_data.at<float>(i, j);
        }
    }

    int no_of_pixel_pad = neighborhood_size / 2;
    float threshold = 5000; // hardcoded threshold, can be parameterized

    // Allocate memory on the GPU
    float* d_img;
    size_t img_size = (cv_mat_data.cols / 2 + 2 * no_of_pixel_pad) * (cv_mat_data.rows / 2 + 2 * no_of_pixel_pad) * sizeof(float);
    cudaMalloc(&d_img, img_size);

    // Define block and grid sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((cv_mat_data.cols / 2 + blockSize.x - 1) / blockSize.x, (cv_mat_data.rows / 2 + blockSize.y - 1) / blockSize.y);

    for (int idx = 0; idx < quarters.size(); ++idx) {
        std::cout << "bad pixel correction: Quarter " << idx + 1 << " of 4" << std::endl;

        cv::Mat img = quarters[idx];
        int width = img.cols;
        int height = img.rows;

        // Pad the image borders and copy to GPU
        cv::Mat padded_img;
        cv::copyMakeBorder(img, padded_img, no_of_pixel_pad, no_of_pixel_pad, no_of_pixel_pad, no_of_pixel_pad, cv::BORDER_REFLECT);
        cudaMemcpy(d_img, padded_img.ptr<float>(), img_size, cudaMemcpyHostToDevice);

        // Launch the kernel
        badPixelCorrectionKernel<<<gridSize, blockSize>>>(d_img, width, height, neighborhood_size, threshold, no_of_pixel_pad);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        }
        // Copy the result back to the host
        cudaMemcpy(padded_img.ptr<float>(), d_img, img_size, cudaMemcpyDeviceToHost);
        // Remove padding and update the quarter
        quarters[idx] = padded_img(cv::Range(no_of_pixel_pad, height + no_of_pixel_pad), cv::Range(no_of_pixel_pad, width + no_of_pixel_pad));
    }

    // Free GPU memory
    cudaFree(d_img);

    // Regroup the quarters back into the original image
    for (int i = 0; i < cv_mat_data.rows; i += 2) {
        for (int j = 0; j < cv_mat_data.cols; j += 2) {
            cv_mat_data.at<float>(i, j) = quarters[0].at<float>(i / 2, j / 2);
            cv_mat_data.at<float>(i, j + 1) = quarters[1].at<float>(i / 2, j / 2);
            cv_mat_data.at<float>(i + 1, j) = quarters[2].at<float>(i / 2, j / 2);
            cv_mat_data.at<float>(i + 1, j + 1) = quarters[3].at<float>(i / 2, j / 2);
        }
    }
}

