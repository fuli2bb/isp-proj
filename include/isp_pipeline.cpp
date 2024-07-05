#include "isp_pipeline.hpp"
#include "utils.hpp"
#include <cmath>
#include "isp_cuda_func.hpp"

typedef std::vector<std::vector<float>> vec_mat;


isp_pipeline::isp_pipeline():
      channel_gain(3), 
      black_level(4), 
      white_level(4), 
      color_matrix(3, std::vector<float>(3, 0)){
}


isp_pipeline::isp_pipeline(const std::string & image_name, ConfigIni& cfg){
  //copy cfg infomation to class member
  name = image_name;
  height =  std::stoi(cfg.getValue(image_name, "height"));
  width =  std::stoi(cfg.getValue(image_name, "width"));
  color_space = cfg.getValue(image_name, "color_space");
  bayer_pattern = cfg.getValue(image_name, "bayer_pattern");
  bit_depth = std::stoi(cfg.getValue(image_name, "bit_depth"));
  channel_gain = parse_vector(cfg.getValue(image_name, "channel_gain"));  
  white_level = parse_vector(cfg.getValue(image_name, "white_level"));
  black_level = parse_vector(cfg.getValue(image_name, "black_level"));
  color_matrix = parse_matrix(cfg.getValue(image_name, "color_matrix"));

  print_image_config();
}

isp_pipeline::~isp_pipeline() {
    // Currently empty because all members are automatically managed.
    // Ready for future expansion if manual resource management becomes necessary.
}



void isp_pipeline::load_raw_data(){
    // load raw and create a cv mat
     raw_data = read_raw_data(name);
    // Check if data is not empty
    if (raw_data.empty()) {
      std::cerr << "Failed to read data from file or file is empty." << std::endl;
      }  
}
    
void isp_pipeline::create_cv_mat_data(){
    cv_mat_data = createmat_from_vector(raw_data, height, width, CV_16U);
    cv_mat_data.convertTo(cv_mat_data, CV_32F);
}
    //cv_mat_data  = &data;



void isp_pipeline::print_image_config() const{
  std::cout << name << "\n";
  std::cout << "Height: " << height << "\n";
  std::cout << "Width: " << width << "\n";
  std::cout << "Color Space: " << color_space << "\n";
  std::cout << "Bayer Pattern: " << bayer_pattern << "\n";
  std::cout << "Bit Depth: " << bit_depth << "\n";

  std::cout << "Channel Gain: ";
  for (float gain : channel_gain) {
    std::cout << gain << " ";
  }
  std::cout << "\n";

  std::cout << "Black Level: ";
  for (int level : black_level) {
    std::cout << level << " ";
  }
  std::cout << "\n";

  std::cout << "White Level: ";
  for (int level : white_level) {
    std::cout << level << " ";
  }
  std::cout << "\n";

  std::cout << "Color Matrix:\n";
  for (const auto& row : color_matrix) {
    for (float value : row) {
      std::cout << value << " ";
    }
    std::cout << "\n";
  }
}


void isp_pipeline::black_level_correction(){
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Running black level correction..." << std::endl;
    
    int maxv =  pow(2,bit_depth)-1;
    int minv = 0;
    // Note: The raw image has a Bayer pattern with four channels (e.g., R, Gr, Gb, B). The correction is applied separately to each channel:
    for (int i = 0; i < cv_mat_data.rows; ++i) {
        for (int j = 0; j < cv_mat_data.cols; ++j) {
            int index = 2 * (i % 2) + (j % 2); // Calculate index for black_level and white_level based on pixel position
            cv_mat_data.at<float>(i, j) = 
            (cv_mat_data.at<float>(i, j) - black_level[index]) / (white_level[index] - black_level[index])*maxv;
        }
    }

    // Clip data to the specified range:
    
    cv_mat_data = cv::max(cv_mat_data, minv);
    cv_mat_data = cv::min(cv_mat_data, maxv);
    double minVal, maxVal;
    cv::minMaxLoc(cv_mat_data, &minVal, &maxVal); 
    std::cout<<minVal<<" "<<maxVal<<std::endl;

}


void isp_pipeline::do_channel_gain_white_balance(){
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Running channel gain white balance..." << std::endl; 

    // Access data pointer for efficient processing
    float* dataPtr = (float*)cv_mat_data.data;
    int rows = cv_mat_data.rows;
    int cols = cv_mat_data.cols;
    //int channels = data.channels();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            // Apply channel gains
            if (i % 2 == 0 && j % 2 == 0) {
                dataPtr[idx] *= channel_gain[0];
            } else if (i % 2 == 0 && j % 2 == 1) {
                dataPtr[idx] *= channel_gain[1];
            } else if (i % 2 == 1 && j % 2 == 0) {
                dataPtr[idx] *= channel_gain[2];
            } else if (i % 2 == 1 && j % 2 == 1) {
                dataPtr[idx] *= channel_gain[3];
            }

            // Clip values within the range
            dataPtr[idx] = std::max(0.0f, dataPtr[idx]);
        }
    }
}

void isp_pipeline::bad_pixel_correction(const int neighborhood_size) {
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Running bad pixel correction..." << std::endl;
    //Method: median filter with threshold
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
    float threshold = 5000; //hardcord XXX
    for (int idx = 0; idx < quarters.size(); ++idx) {
        std::cout << "bad pixel correction: Quarter " << idx + 1 << " of 4" << std::endl;

        cv::Mat img = quarters[idx];
        int width = img.cols;
        int height = img.rows;

        // Pad the image borders
        cv::copyMakeBorder(img, img, no_of_pixel_pad, no_of_pixel_pad, no_of_pixel_pad, no_of_pixel_pad, cv::BORDER_REFLECT);
        for (int i = no_of_pixel_pad; i < height + no_of_pixel_pad; ++i) {
            for (int j = no_of_pixel_pad; j < width + no_of_pixel_pad; ++j) {
                float mid_pixel_val = img.at<float>(i, j);

                // Extract the neighborhood into a vector
                std::vector<float> neighborhood_vec;
                for (int ni = -no_of_pixel_pad; ni <= no_of_pixel_pad; ++ni) {
                    for (int nj = -no_of_pixel_pad; nj <= no_of_pixel_pad; ++nj) {
                        neighborhood_vec.push_back(img.at<float>(i + ni, j + nj));
                    }
                }
                // Calculate the median of the neighborhood
                std::nth_element(neighborhood_vec.begin(), neighborhood_vec.begin() + neighborhood_vec.size() / 2, neighborhood_vec.end());
                float median_val = neighborhood_vec[neighborhood_vec.size() / 2];

                // If the middle pixel value deviates significantly from the median value,
                // consider it as a bad pixel and replace it with the median value
                if (std::abs(mid_pixel_val - median_val) > threshold) {
                    img.at<float>(i, j) = median_val;
                }
            }
        }

        // Remove padding and update the quarter
        quarters[idx] = img(cv::Range(no_of_pixel_pad, height + no_of_pixel_pad), cv::Range(no_of_pixel_pad, width + no_of_pixel_pad));
    }
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




vec_mat initializeMatrix(int rows, int cols, float value = 0.0f) {
    return vec_mat(rows, std::vector<float>(cols, value));
}

// Function to calculate gradients
void calculateGradients(const cv::Mat& data, vec_mat& v, vec_mat& h) {
    int rows = data.rows;
    int cols = data.cols;

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            v[i][j] = data.at<float>(i - 1, j) - data.at<float>(i + 1, j);
            h[i][j] = data.at<float>(i, j - 1) - data.at<float>(i, j + 1);
        }
    }
    // Reflect borders to match the 'symm' boundary condition
    for (int j = 1; j < cols - 1; ++j) {
        v[0][j] = data.at<float>(1, j) - data.at<float>(0, j);
        v[rows - 1][j] = data.at<float>(rows - 2, j) - data.at<float>(rows - 1, j);
    }
    for (int i = 1; i < rows - 1; ++i) {
        h[i][0] = data.at<float>(i, 1) - data.at<float>(i, 0);
        h[i][cols - 1] = data.at<float>(i, cols - 2) - data.at<float>(i, cols - 1);
    }
}

void fillG_directional(const cv::Mat& data, cv::Mat& output, const std::string& bayer_pattern) {
    int rows = data.rows;
    int cols = data.cols;

    vec_mat v = initializeMatrix(rows, cols);
    vec_mat h = initializeMatrix(rows, cols);
    calculateGradients(data, v, h);

    vec_mat weight_N = initializeMatrix(rows, cols);
    vec_mat weight_E = initializeMatrix(rows, cols);
    vec_mat weight_S = initializeMatrix(rows, cols);
    vec_mat weight_W = initializeMatrix(rows, cols);

    vec_mat value_N = initializeMatrix(rows, cols);
    vec_mat value_E = initializeMatrix(rows, cols);
    vec_mat value_S = initializeMatrix(rows, cols);
    vec_mat value_W = initializeMatrix(rows, cols);

    if (bayer_pattern == "rggb" || bayer_pattern == "bggr") {
        // Calculate weights and values for blue (B) locations
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                if (i > 0 && j < cols - 1) weight_N[i][j] = std::abs(v[i][j]) + std::abs(v[i - 1][j]);
                if (j < cols - 1) weight_E[i][j] = std::abs(h[i][j]) + std::abs(h[i][j + 1]);
                if (i < rows - 1) weight_S[i][j] = std::abs(v[i][j]) + std::abs(v[i + 1][j]);
                if (j > 0) weight_W[i][j] = std::abs(h[i][j]) + std::abs(h[i][j - 1]);

                if (i > 0) value_N[i][j] = data.at<float>(i - 1, j) + v[i - 1][j] / 2.0f;
                if (j < cols - 1) value_E[i][j] = data.at<float>(i, j + 1) - h[i][j + 1] / 2.0f;
                if (i < rows - 1) value_S[i][j] = data.at<float>(i + 1, j) - v[i + 1][j] / 2.0f;
                if (j > 0) value_W[i][j] = data.at<float>(i, j - 1) + h[i][j - 1] / 2.0f;
            }
        }

        // Calculate weights and values for red (R) locations
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                if (i > 0 && j < cols - 1) weight_N[i][j] = std::abs(v[i][j]) + std::abs(v[i - 1][j]);
                if (j < cols - 1) weight_E[i][j] = std::abs(h[i][j]) + std::abs(h[i][j + 1]);
                if (i < rows - 1) weight_S[i][j] = std::abs(v[i][j]) + std::abs(v[i + 1][j]);
                if (j > 0) weight_W[i][j] = std::abs(h[i][j]) + std::abs(h[i][j - 1]);

                if (i > 0) value_N[i][j] = data.at<float>(i - 1, j) + v[i - 1][j] / 2.0f;
                if (j < cols - 1) value_E[i][j] = data.at<float>(i, j + 1) - h[i][j + 1] / 2.0f;
                if (i < rows - 1) value_S[i][j] = data.at<float>(i + 1, j) - v[i + 1][j] / 2.0f;
                if (j > 0) value_W[i][j] = data.at<float>(i, j - 1) + h[i][j - 1] / 2.0f;
            }
        }

        // Normalize weights
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                weight_N[i][j] = 1.0f / (1.0f + weight_N[i][j]);
                weight_E[i][j] = 1.0f / (1.0f + weight_E[i][j]);
                weight_S[i][j] = 1.0f / (1.0f + weight_S[i][j]);
                weight_W[i][j] = 1.0f / (1.0f + weight_W[i][j]);
            }
        }

        // Calculate the output
        output = cv::Mat::zeros(rows, cols, CV_32F);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float sum_weights = weight_N[i][j] + weight_E[i][j] + weight_S[i][j] + weight_W[i][j];
                float interpolated_value = (value_N[i][j] * weight_N[i][j] +
                                            value_E[i][j] * weight_E[i][j] +
                                            value_S[i][j] * weight_S[i][j] +
                                            value_W[i][j] * weight_W[i][j]) / sum_weights;
                output.at<float>(i, j) = std::max(0.0f,interpolated_value);
            }
        }
        // Copy original values at certain locations
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                output.at<float>(i, j) = data.at<float>(i, j);
            }
        }
        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                output.at<float>(i, j) = data.at<float>(i, j);
            }
        }
    }
}


void fillG_bilinear(const cv::Mat& data, cv::Mat& G, const std::string& bayer_pattern) {
    int rows = data.rows;
    int cols = data.cols;

    // Ensure the bayer pattern is 'rggb'
    if (bayer_pattern != "rggb") {
        std::cerr << "Unsupported Bayer pattern!" << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if ((i % 2 == 0) && (j % 2 == 0)) { // Red pixel
                G.at<float>(i, j) = ((i > 0 ? data.at<float>(i - 1, j) : 0) + (i < rows - 1 ? data.at<float>(i + 1, j) : 0) +
                                     (j > 0 ? data.at<float>(i, j - 1) : 0) + (j < cols - 1 ? data.at<float>(i, j + 1) : 0)) / 4.0f;
            } else if ((i % 2 == 0) && (j % 2 == 1)) { // Green pixel (on red row)
                G.at<float>(i, j) = data.at<float>(i, j);
            } else if ((i % 2 == 1) && (j % 2 == 0)) { // Green pixel (on blue row)
                G.at<float>(i, j) = data.at<float>(i, j);
            } else { // Blue pixel
                G.at<float>(i, j) = ((i > 0 ? data.at<float>(i - 1, j) : 0) + (i < rows - 1 ? data.at<float>(i + 1, j) : 0) +
                                     (j > 0 ? data.at<float>(i, j - 1) : 0) + (j < cols - 1 ? data.at<float>(i, j + 1) : 0)) / 4.0f;
            }
        }
    }
}

void fillRB_bilinear(const cv::Mat& data, cv::Mat& R, cv::Mat& B, const std::string& bayer_pattern){

   int rows = data.rows;
   int cols = data.cols;
   // for rggb only
   for (int i = 0; i < rows; ++i) {
       for (int j = 0; j < cols; ++j) {
           if ((i % 2 == 0) && (j % 2 == 0)) { // Red pixel
                R.at<float>(i, j) = data.at<float>(i, j);
                B.at<float>(i, j) = ((i > 0 && j > 0 ? data.at<float>(i - 1, j - 1) : 0) + (i > 0 && j < cols - 1 ? data.at<float>(i - 1, j + 1) : 0) +
                     (i < rows - 1 && j > 0 ? data.at<float>(i + 1, j - 1) : 0) + (i < rows - 1 && j < cols - 1 ? data.at<float>(i + 1, j + 1) : 0)) / 4.0f;
            } else if ((i % 2 == 0) && (j % 2 == 1)) { // Green pixel (on red row)
                R.at<float>(i, j) = ((j > 0 ? data.at<float>(i, j - 1) : 0) + (j < cols - 1 ? data.at<float>(i, j + 1) : 0)) / 2.0f;
                B.at<float>(i, j) = ((i > 0 ? data.at<float>(i - 1, j) : 0) + (i < rows - 1 ? data.at<float>(i + 1, j) : 0)) / 2.0f;
            } else if ((i % 2 == 1) && (j % 2 == 0)) { // Green pixel (on blue row)
                R.at<float>(i, j) = ((i > 0 ? data.at<float>(i - 1, j) : 0) + (i < rows - 1 ? data.at<float>(i + 1, j) : 0)) / 2.0f;
                B.at<float>(i, j) = ((j > 0 ? data.at<float>(i, j - 1) : 0) + (j < cols - 1 ? data.at<float>(i, j + 1) : 0)) / 2.0f;
            } else { // Blue pixel
                B.at<float>(i, j) = data.at<float>(i, j);
                R.at<float>(i, j) = ((i > 0 && j > 0 ? data.at<float>(i - 1, j - 1) : 0) + (i > 0 && j < cols - 1 ? data.at<float>(i - 1, j + 1) : 0) +
                     (i < rows - 1 && j > 0 ? data.at<float>(i + 1, j - 1) : 0) + (i < rows - 1 && j < cols - 1 ? data.at<float>(i + 1, j + 1) : 0)) / 4.0f;
            }

        }
    }
}



void isp_pipeline::do_demosaic(){

  std::cout << "----------------------------------------------------" << std::endl;
  std::cout << "Running demosaic procedure..." << std::endl;


  cv::Mat R = cv::Mat::zeros(cv_mat_data.size(), CV_32F);
  cv::Mat G = cv::Mat::zeros(cv_mat_data.size(), CV_32F);
  cv::Mat B = cv::Mat::zeros(cv_mat_data.size(), CV_32F);

  fillG_directional(cv_mat_data, G, bayer_pattern); 
  fillRB_bilinear(cv_mat_data, R, B, bayer_pattern);
  //fillG_bilinear(cv_mat_data, G, bayer_pattern);

  std::vector<cv::Mat> channels = {B, G, R};
  cv::merge(channels, cv_mat_data);

}

cv::Mat convolve(const cv::Mat& src, const cv::Mat& kernel){
    cv::Mat dst;
    cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    return dst;
}

void isp_pipeline::post_process_local_color_ratio(float beta){
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Demosaicing post process using local color ratio..." << std::endl;

    int rows = cv_mat_data.rows;
    int cols = cv_mat_data.cols;

    // Add beta to the data to prevent divide by zero
    cv::Mat data_beta;
    cv_mat_data.convertTo(data_beta, -1, 1.0, beta); // data_beta = data + beta

    // Convolution kernels
    cv::Mat zeta1 = (cv::Mat_<float>(3, 3) << 0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0);
    cv::Mat zeta2 = (cv::Mat_<float>(3, 3) << 0.25, 0, 0.25, 0, 0, 0, 0.25, 0, 0.25);

    // Compute average of color ratios

        // Split channels
    std::vector<cv::Mat> channels(3);
    cv::split(data_beta, channels);

    // Compute average of color ratios
    cv::Mat g_over_b = convolve(channels[1] / channels[2], zeta1);
    cv::Mat g_over_r = convolve(channels[1] / channels[0], zeta1);
    cv::Mat b_over_g_zeta2 = convolve(channels[2] / channels[1], zeta2);
    cv::Mat r_over_g_zeta2 = convolve(channels[0] / channels[1], zeta2);
    cv::Mat b_over_g_zeta1 = convolve(channels[2] / channels[1], zeta1);
    cv::Mat r_over_g_zeta1 = convolve(channels[0] / channels[1], zeta1);


    if (bayer_pattern == "rggb") {
        // G at B locations
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                cv_mat_data.at<cv::Vec3f>(i, j)[1] = -beta + data_beta.at<cv::Vec3f>(i, j)[2] * g_over_b.at<float>(i, j);
            }
        }

        // G at R locations
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                cv_mat_data.at<cv::Vec3f>(i, j)[1] = -beta + data_beta.at<cv::Vec3f>(i, j)[0] * g_over_r.at<float>(i, j);
            }
        }

        // B at R locations
        for (int i = 0; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                cv_mat_data.at<cv::Vec3f>(i, j)[2] = -beta + data_beta.at<cv::Vec3f>(i, j)[1] * b_over_g_zeta2.at<float>(i, j);
            }
        }

        // R at B locations
        for (int i = 1; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                cv_mat_data.at<cv::Vec3f>(i, j)[0] = -beta + data_beta.at<cv::Vec3f>(i, j)[1] * r_over_g_zeta2.at<float>(i, j);
            }
        }

        // B at G locations
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                cv_mat_data.at<cv::Vec3f>(i, j)[2] = -beta + data_beta.at<cv::Vec3f>(i, j)[1] * b_over_g_zeta1.at<float>(i, j);
            }
        }

        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                cv_mat_data.at<cv::Vec3f>(i, j)[2] = -beta + data_beta.at<cv::Vec3f>(i, j)[1] * b_over_g_zeta1.at<float>(i, j);
            }
        }
        // R at G locations
        for (int i = 0; i < rows; i += 2) {
            for (int j = 1; j < cols; j += 2) {
                cv_mat_data.at<cv::Vec3f>(i, j)[0] = -beta + data_beta.at<cv::Vec3f>(i, j)[1] * r_over_g_zeta1.at<float>(i, j);
            }
        }

        for (int i = 1; i < rows; i += 2) {
            for (int j = 0; j < cols; j += 2) {
                cv_mat_data.at<cv::Vec3f>(i, j)[0] = -beta + data_beta.at<cv::Vec3f>(i, j)[1] * r_over_g_zeta1.at<float>(i, j);
            }
        }
    }
    cv::max(cv_mat_data, 0, cv_mat_data);
    // Additional conditions for other Bayer patterns (grbg, gbrg, bggr) can be implemented similarly
}


cv::Mat& isp_pipeline::get_cv_mat(){
    return cv_mat_data;
}

const vec_mat& get_rgb2xyz(const std::string& color_space, const std::string& illuminant) {
    static const vec_mat srgb_d65 = {{.4124564, .3575761, .1804375},
                                     {.2126729, .7151522, .0721750},
                                     {.0193339, .1191920, .9503041}};
    static const vec_mat srgb_d50 = {{.4360747, .3850649, .1430804},
                                     {.2225045, .7168786, .0606169},
                                     {.0139322, .0971045, .7141733}};
    static const vec_mat adobe_rgb_d65 = {{.5767309, .1855540, .1881852},
                                          {.2973769, .6273491, .0752741},
                                          {.0270343, .0706872, .9911085}};
    static const vec_mat adobe_rgb_d50 = {{.6097559, .2052401, .1492240},
                                          {.3111242, .6256560, .0632197},
                                          {.0194811, .0608902, .7448387}};
    static const vec_mat empty;

    if (color_space == "srgb") {
        if (illuminant == "d65") {
            return srgb_d65;
        } else if (illuminant == "d50") {
            return srgb_d50;
        } else {
            std::cerr << "for now, illuminant must be d65 or d50" << std::endl;
            return empty;
        }
    } else if (color_space == "adobe-rgb-1998") {
        if (illuminant == "d65") {
            return adobe_rgb_d65;
        } else if (illuminant == "d50") {
            return adobe_rgb_d50;
        } else {
            std::cerr << "for now, illuminant must be d65 or d50" << std::endl;
            return empty;
        }
    } else {
        std::cerr << "for now, color_space must be srgb or adobe-rgb-1998" << std::endl;
        return empty;
    }
}

cv::Mat calculate_cam2rgb(const vec_mat& xyz2cam, const vec_mat& rgb2xyz) {
    // Convert vec_mat to cv::Mat
    cv::Mat rgb2cam(3, 3, CV_32F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            rgb2cam.at<float>(i, j) = xyz2cam[i][0] * rgb2xyz[0][j] +
                                      xyz2cam[i][1] * rgb2xyz[1][j] +
                                      xyz2cam[i][2] * rgb2xyz[2][j];

    // Normalize rows to sum to 1
    
    
    
    for (int i = 0; i < 3; ++i) {
        float row_sum = cv::sum(rgb2cam.row(i))[0];
        if (row_sum != 0) {
            rgb2cam.row(i) /= row_sum;
        }
    }

    // Check if the matrix is invertible
    cv::Mat cam2rgb;
    if (cv::determinant(rgb2cam) != 0) {
        cam2rgb = rgb2cam.inv();
    } else {
        std::cerr << "Warning! matrix not invertible." << std::endl;
        cam2rgb = cv::Mat::eye(3, 3, CV_32F);
    }
    return cam2rgb;
}

void isp_pipeline::apply_cmatrix() {
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Running color correction..." << std::endl;

    // Check if data is 3 dimensional
    if (cv_mat_data.channels() != 3) {
        std::cerr << "Data needs to be three dimensional" << std::endl;
        return;
    }
    
    std::string illuminant = "d65";
    // Get the color correction matrix
    vec_mat rgb2xyz = get_rgb2xyz(color_space, illuminant);
    if (rgb2xyz.empty()) {
        std::cerr << "Invalid color space or illuminant" << std::endl;
        return;
    }

    cv::Mat cam2rgb = calculate_cam2rgb(color_matrix, rgb2xyz);

    // Split the channels
    std::vector<cv::Mat> channels(3);
    cv::split(cv_mat_data, channels);

    cv::Mat R = channels[2];
    cv::Mat G = channels[1];
    cv::Mat B = channels[0];

    // Apply the matrix
    std::vector<cv::Mat> corrected_channels(3);
    corrected_channels[2] = cv::max(0.0f, R * cam2rgb.at<float>(0, 0) + G * cam2rgb.at<float>(0, 1) + B * cam2rgb.at<float>(0, 2));
    corrected_channels[1] = cv::max(0.0f, R * cam2rgb.at<float>(1, 0) + G * cam2rgb.at<float>(1, 1) + B * cam2rgb.at<float>(1, 2));
    corrected_channels[0] = cv::max(0.0f, R * cam2rgb.at<float>(2, 0) + G * cam2rgb.at<float>(2, 1) + B * cam2rgb.at<float>(2, 2));

    // Merge the corrected channels back into cv_mat_data
    cv::merge(corrected_channels, cv_mat_data);
}



void isp_pipeline::do_gamma(){
    float ratio = 80;
    float a = -0.9;
    float b =-8.0;
    float clip_max = 65535.0;
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Running brightening..." << std::endl;
    
    cv_mat_data =  cv_mat_data*std::log10(ratio);

    cv::min(cv_mat_data, clip_max, cv_mat_data);
    cv::max(cv_mat_data, 0.0f, cv_mat_data);
    
    std::cout<< "----------------------------------------------------" <<std::endl;
    std::cout<< "Running nonlinearity by equation..." << std::endl;
    // Normalize the data to the range [0, 1]
    //cv::Mat data_float;
    //cv_mat_data.convertTo(data_float, CV_32F);

    cv_mat_data /= clip_max;

    // Apply the nonlinearity in place
    cv::exp(b * cv_mat_data, cv_mat_data); // cv_mat_data = exp(b * cv_mat_data)
    cv_mat_data = clip_max * (a * cv_mat_data + cv_mat_data + a * cv_mat_data - a * std::exp(b) * cv_mat_data - a);

    // Clip the result to the specified range
    cv::min(cv_mat_data, clip_max, cv_mat_data);
    cv::max(cv_mat_data, 0.0f, cv_mat_data);
}



void nonuniform_quantization(const cv::Mat& channel, cv::Mat& output) {
    output = cv::Mat::zeros(channel.size(), CV_32F);
    double min_val, max_val;
    cv::minMaxLoc(channel, &min_val, &max_val);

    float range = max_val - min_val;
    float threshold_78 = (7.0 / 8.0) * range;
    float threshold_34 = (3.0 / 4.0) * range;
    float threshold_12 = (1.0 / 2.0) * range;

    for (int i = 0; i < channel.rows; ++i) {
        for (int j = 0; j < channel.cols; ++j) {
            float value = channel.at<float>(i, j);
            if (value > threshold_78) {
                output.at<float>(i, j) = 3.0;
            } else if (value > threshold_34) {
                output.at<float>(i, j) = 2.0;
            } else if (value > threshold_12) {
                output.at<float>(i, j) = 1.0;
            }
        }
    }
}

void sobel_gradient_magnitude(const cv::Mat& channel, int ksize, cv::Mat& output) {
    cv::Mat grad_x, grad_y;
    cv::Sobel(channel, grad_x, CV_32F, 1, 0, ksize);
    cv::Sobel(channel, grad_y, CV_32F, 0, 1, ksize);

    cv::magnitude(grad_x, grad_y, output);
}


void isp_pipeline::purple_fringe_removal(float nsr_threshold, float cr_threshold) {
    int width = cv_mat_data.cols;
    int height = cv_mat_data.rows;

    std::vector<cv::Mat> channels(3);
    cv::split(cv_mat_data, channels);

    cv::Mat r = channels[2];
    cv::Mat g = channels[1];
    cv::Mat b = channels[0];

    nsr_threshold = 65535.0* nsr_threshold / 100;
    cv::Mat temp = (r + g + b) / 3.0;

    cv::Mat nsr;
    cv::threshold(temp, nsr, nsr_threshold, 1, cv::THRESH_BINARY);

    cv::Mat temp_r_b = r - b;
    cv::Mat temp_b_g = b - g;
    cv::Mat cr = ((temp_r_b < cr_threshold) & (temp_b_g > cr_threshold)) / 255;
    cr.convertTo(cr, CV_32F);

    cv::Mat qr, qg, qb;
    nonuniform_quantization(r, qr);
    nonuniform_quantization(g, qg);
    nonuniform_quantization(b, qb);

    cv::Mat g_qr, g_qg, g_qb;
    sobel_gradient_magnitude(qr, 5, g_qr);
    sobel_gradient_magnitude(qg, 5, g_qg);
    sobel_gradient_magnitude(qb, 5, g_qb);

    cv::Mat bgm = (g_qr != 0) | (g_qg != 0) | (g_qb != 0);
    bgm.convertTo(bgm, CV_32F);

    cv::Mat fringe_map = nsr.mul(cr).mul(bgm);
    cv::Mat mask = (fringe_map == 1.0);

    cv::Mat r1 = r.clone();
    cv::Mat g1 = g.clone();
    cv::Mat b1 = b.clone();
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            if (mask.at<float>(i, j) == 1.0f) {
                float avg = (r.at<float>(i, j) + g.at<float>(i, j) + b.at<float>(i, j)) / 3.0f;
                r1.at<float>(i, j) = avg;
                g1.at<float>(i, j) = avg;
                b1.at<float>(i, j) = avg;
            }
        }
    }

    std::vector<cv::Mat> result_channels = {b1, g1, r1};
    cv::merge(result_channels, cv_mat_data);
}



cv::Mat createGaussianKernel(cv::Size kernelSize, double sigma) {
    cv::Mat kernelX = cv::getGaussianKernel(kernelSize.width, sigma, CV_32F);
    cv::Mat kernelY = cv::getGaussianKernel(kernelSize.height, sigma, CV_32F);
    cv::Mat kernel = kernelX * kernelY.t();
    return kernel;
}

// Function for soft coring
cv::Mat softCoring(const cv::Mat& highPass, double slope, double tauThreshold, double gammaSpeed) {
    cv::Mat absHighPass;
    cv::absdiff(highPass, cv::Scalar::all(0), absHighPass);

    cv::Mat coreTerm;
    cv::pow(absHighPass / tauThreshold, gammaSpeed, coreTerm);
    cv::exp(-coreTerm, coreTerm);
    coreTerm = slope * highPass.mul(1.0 - coreTerm);
    return coreTerm;
}


void isp_pipeline::do_tone_mapping(){
    
    double strength_multiplier = 1.0;
    double gaussian_sigma = 1.0;
    cv::Size gaussian_kernel_size = {5,5};

    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Running tone mapping by non linear masking..." << std::endl;

    cv::Mat gray_image;

    // Convert to gray image
    if (cv_mat_data.channels() == 3) {
        cv::cvtColor(cv_mat_data, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = cv_mat_data;
    }

    // Gaussian blur the gray image
    cv::Mat gaussian_kernel = createGaussianKernel(gaussian_kernel_size, gaussian_sigma);
    cv::Mat mask;
    cv::filter2D(gray_image, mask, -1, gaussian_kernel);

    // Bring the mask within range 0 to 1 and multiply with strength_multiplier
    mask = strength_multiplier * mask / 65535.0;

    // Calculate the alpha image
    cv::Mat temp;
    cv::exp(-mask * std::log(2.0), temp);  // This is equivalent to 0.5^mask

    cv::Mat alpha;
    if (cv_mat_data.channels() == 3) {
        std::vector<cv::Mat> channels(3, temp);
        cv::merge(channels, alpha);
    } else {
        alpha = temp;
    }

    // Perform tone mapping
    cv::Mat normalized_data = cv_mat_data / 65535.0;
    cv::Mat result = normalized_data.clone();

    for (int i = 0; i < normalized_data.rows; ++i) {
        for (int j = 0; j < normalized_data.cols; ++j) {
            if (normalized_data.channels() == 3) {
                for (int c = 0; c < 3; ++c) {
                    result.at<cv::Vec3f>(i, j)[c] = std::pow(normalized_data.at<cv::Vec3f>(i, j)[c], alpha.at<cv::Vec3f>(i, j)[c]);
                }
            } else {
                result.at<float>(i, j) = std::pow(normalized_data.at<float>(i, j), alpha.at<float>(i, j));
            }
        }
    }

    result = 65535.0 * result;
    cv::min(result,  65535.0, result);
    cv::max(result, 0.0, result);

    cv_mat_data = result;
}

void isp_pipeline::do_sharpening(){
    cv::Size gaussianKernelSize = {5, 5};
    double gaussianSigma = 2.0;
    double slope = 1.5; 
    double tauThreshold = 0.05; 
    double gammaSpeed = 4.0; 
    double clipMin = 0; 
    double clipMax = 65535;

    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Running sharpening by unsharp masking..." << std::endl;

    // Create Gaussian kernel
    cv::Mat gaussianKernel = createGaussianKernel(gaussianKernelSize, gaussianSigma);
    // Convolve the image with the Gaussian kernel
    cv::Mat imageBlur;
    cv::filter2D(cv_mat_data, imageBlur, -1, gaussianKernel);

    // High frequency component image
    cv::Mat imageHighPass = cv_mat_data - imageBlur;
    // Soft coring
    tauThreshold *= clipMax;
    cv::Mat softCoredHighPass = softCoring(imageHighPass, slope, tauThreshold, gammaSpeed);

    // Add the soft cored high pass image to the original and clip within range
    cv_mat_data += softCoredHighPass;
    cv::min(cv_mat_data, clipMax, cv_mat_data);
    cv::max(cv_mat_data, clipMin, cv_mat_data);
}


cv::Mat isp_pipeline::rgb2xyz(const cv::Mat& img) {
    cv::Mat imgXYZ;
    cv::cvtColor(img, imgXYZ, cv::COLOR_BGR2XYZ);
    return imgXYZ;
}

// Helper function for XYZ to LAB conversion
cv::Mat isp_pipeline::xyz2lab(const cv::Mat& img) {
    cv::Mat imgLAB;
    cv::cvtColor(img, imgLAB, cv::COLOR_BGR2Lab);
    return imgLAB;
}

// Helper function for LAB to LCH conversion
cv::Mat isp_pipeline::lab2lch(const cv::Mat& imgLAB) {
    cv::Mat imgLCH = imgLAB.clone();
    for (int i = 0; i < imgLAB.rows; ++i) {
        for (int j = 0; j < imgLAB.cols; ++j) {
            cv::Vec3f& lab = imgLCH.at<cv::Vec3f>(i, j);
            float L = lab[0], a = lab[1], b = lab[2];
            float C = sqrt(a * a + b * b);
            float H = atan2(b, a) * 180 / CV_PI;
            if (H < 0) H += 360;
            lab[1] = C;
            lab[2] = H;
        }
    }
    return imgLCH;
}

// Helper function for LCH to LAB conversion
cv::Mat isp_pipeline::lch2lab(const cv::Mat& imgLCH) {
    cv::Mat imgLAB = imgLCH.clone();
    for (int i = 0; i < imgLCH.rows; ++i) {
        for (int j = 0; j < imgLCH.cols; ++j) {
            cv::Vec3f& lch = imgLAB.at<cv::Vec3f>(i, j);
            float L = lch[0], C = lch[1], H = lch[2] * CV_PI / 180;
            float a = C * cos(H);
            float b = C * sin(H);
            lch[1] = a;
            lch[2] = b;
        }
    }
    return imgLAB;
}


cv::Mat isp_pipeline::lab2xyz(const cv::Mat& imgLAB) {
    cv::Mat imgXYZ;
    cv::cvtColor(imgLAB, imgXYZ, cv::COLOR_Lab2BGR);
    cv::cvtColor(imgXYZ, imgXYZ, cv::COLOR_BGR2XYZ);
    return imgXYZ;
}
// Helper function for XYZ to RGB conversion
cv::Mat isp_pipeline::xyz2rgb(const cv::Mat& imgXYZ) {
    cv::Mat imgRGB;
    cv::cvtColor(imgXYZ, imgRGB, cv::COLOR_XYZ2BGR);
    return imgRGB;
}
void customEqualizeHist(cv::Mat& channel) {
    // Calculate histogram
    int histSize = 65536;
    std::vector<int> hist(histSize, 0);
    for (int y = 0; y < channel.rows; ++y) {
        for (int x = 0; x < channel.cols; ++x) {
            hist[channel.at<ushort>(y, x)]++;
        }
    }
    // Calculate cumulative distribution function (CDF)
    std::vector<double> cdf(histSize, 0);
    cdf[0] = hist[0];
    for (int i = 1; i < histSize; ++i) {
        cdf[i] = cdf[i - 1] + hist[i];
        //if (i%10000==0) std::cout<<cdf[i]<<" ";
        //cout<<cdf[i]<<" ";
    }
    //std::cout<<"\n";
    // Normalize CDF to [0, 65535]
    double cdfMin = cdf[0];
    double totalPixels = cdf[histSize - 1];
    for (int i = 0; i < histSize; ++i) {
        cdf[i] = ((cdf[i] - cdfMin) / (totalPixels - cdfMin)) * 65535.0;
    }
    // Equalize the image
    for (int y = 0; y < channel.rows; ++y) {
        for (int x = 0; x < channel.cols; ++x) {
            int val = channel.at<ushort>(y, x);
            channel.at<ushort>(y, x) = cv::saturate_cast<ushort>(cdf[val]);
        }
    }
}

void isp_pipeline::equalizeHistColor_opencv() {
    // Ensure the image type is CV_32F
    //printMatType(cv_mat_data);
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
        customEqualizeHist(channels[i]);
    }

    // Merge the channels back
    cv::merge(channels, cv_mat_data);

    // Convert back to CV_32F
    cv_mat_data.convertTo(cv_mat_data, CV_32FC3);
}

template<typename T>
void isp_pipeline::pipeline_opencv(T& image, const std::string& encoding) {
}

#ifdef HAS_CUDA
template<typename T>
void isp_pipeline::pipeline_opencv_gpu(T& image, const std::string& encoding) {
}
#endif

template<typename T>
void isp_pipeline::pipeline(T& image, const std::string& encoding) {
}

