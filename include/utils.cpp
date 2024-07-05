#include "utils.hpp"

std::vector<uint16_t> read_raw_data(const std::string& file_name){
    std::ifstream file("./data/"+file_name+".raw", std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file at " << file_name << std::endl;
        return {}; // Return an empty vector on failure
    }

    // Determine the file size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the data
    std::vector<uint16_t> buffer(size / sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    if (!file) {
        std::cerr << "Error reading file at " << file_name << std::endl;
        return {}; // Return an empty vector on failure
    }

    file.close();
    return buffer;
}


void printMatType(const cv::Mat& mat) {
    int type = mat.type();

    // Depth of the matrix (data type)
    int depth = type & CV_MAT_DEPTH_MASK;
    // Number of channels
    int channels = 1 + (type >> CV_CN_SHIFT);

    std::string r;

    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (channels + '0');

    std::cout << "Matrix type: " << r << std::endl;
}


void imsave(const cv::Mat& input, const std::string& output_name) {
    cv::Mat data = input.clone();
    std::string output_file_type = output_name.substr(output_name.find_last_of(".") + 1);

    // Handle scaling if requested
    //double minVal, maxVal;
    //cv::minMaxLoc(data, &minVal, &maxVal);
    //std::cout<<"Saving data: "<<minVal<<" "<<maxVal<<std::endl;
    // Scale data to the range of the type T
    data.convertTo(data, CV_16U, 1.0f, 0.0f);
    //data.convertTo(data, CV_16U, 65535.0 / (maxVal - minVal), -minVal / (maxVal - minVal));

    // Save the image according to the file type
    if (output_file_type == "png" || output_file_type == "jpg") {
        // Check if it's a grayscale or RGB image
        if (data.channels() == 1) {
            // Single-channel grayscale image
            cv::imwrite(output_name, data);
        } else if (data.channels() == 3) {
            // Three-channel RGB image
            std::vector<cv::Mat> channels(3);
            cv::split(data, channels); // Split channels: BGR order
            cv::imwrite(output_name, data); // Save RGB image
        } else {
            std::cerr << "Unsupported number of channels. Supported are 1 (grayscale) and 3 (RGB)." << std::endl;
        }
    } else {
        std::cerr << "Unsupported file format. Supported formats are .png and .jpg." << std::endl;
    }
}






std::vector<float> parse_vector(const std::string& input) {
    std::vector<float> result;

    std::string trimmed = input.substr(1, input.size() - 2);
    std::stringstream ss(trimmed);
    std::string token;

    while (getline(ss, token, ',')) {
        result.push_back(std::stof(token));
    }
    return result;
}



std::vector<std::vector<float>> parse_matrix(const std::string& input) {
    std::vector<std::vector<float>> matrix;
    std::string trimmed = input.substr(1, input.size() - 2);
    std::stringstream ss(trimmed);
    std::string row;
    
    while (getline(ss, row, ']')) {
        if (row[0] == '[') { 
            row = row.substr(1);
        }
        if (row[0] == ',') {
            row = row.substr(1);
        }
        if (!row.empty()) {
            matrix.push_back(parse_vector(row));
        }
    }
    
    return matrix;
}



