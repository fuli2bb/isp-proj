#include <iostream>
#include <opencv2/opencv.hpp>
#include <CL/cl.hpp>

int main() {
    cv::Mat image = cv::imread("/home/fuli2/sosf.png");
    if (image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }
    cv::imshow("Loaded Image", image);
    cv::waitKey(0);

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device = devices.front();

    cl::Context context(device);
    cl::Buffer clBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

    std::cout << "OpenCL buffer created successfully." << std::endl;

    return 0;
}

