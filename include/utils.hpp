#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp> 

std::vector<uint16_t> read_raw_data(const std::string& file_name);


template <typename T>
cv::Mat createmat_from_vector(std::vector<T>& vec, int rows, int cols, int type) {
    return cv::Mat(rows, cols, type, vec.data()).clone();
}


void printMatType(const cv::Mat& mat);

void imsave(const cv::Mat& input, const std::string& output_name);

std::vector<float> parse_vector(const std::string& input);

std::vector<std::vector<float>> parse_matrix(const std::string& input);




#endif // UTILS_HPP
