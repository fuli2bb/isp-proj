#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <string>
#include <vector>

#ifdef HAS_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif
#include "confini.hpp"


class isp_pipeline{
    public:
        isp_pipeline();
        isp_pipeline(const std::string& image_name, ConfigIni & cfg);
        ~isp_pipeline();
        
        void load_raw_data();
        
        void black_level_correction();
        
        void bad_pixel_correction(const int neighorhood_size);
        
        void bad_pixel_correction_cuda(const int neighorhood_size);

        void do_channel_gain_white_balance();

        void create_cv_mat_data();

        void do_demosaic();

        void post_process_local_color_ratio(const float beta);

        void do_gamma();

        void apply_cmatrix();

        void purple_fringe_removal(float nsr_threshold, float cr_threshold);

        void do_sharpening();
        
        void do_tone_mapping();
        
        void equalizeHistColor_opencv();
        
        void equalizeHistColor_cuda();
        
        template< typename T>
        void pipeline_opencv(T& image, const std::string& encoding);
#ifdef HAS_CUDA
        template< typename T>
        void pipeline_opencv_gpu(T& image, const std::string& encoding);
#endif     
        template< typename T>
        void pipeline(T& image, const std::string& encoding);


        cv::Mat& get_cv_mat();



    private:
        std::string name;
        int height;
        int width;
        std::string color_space;
        std::string bayer_pattern;
        int bit_depth;
        std::vector<float> channel_gain;
        std::vector<float> black_level;
        std::vector<float> white_level;
        std::vector<std::vector<float>> color_matrix;
        
        cv::Mat cv_mat_data;
        std::vector<uint16_t> raw_data;
        void print_image_config() const;
        cv::Mat lch2lab(const cv::Mat& imgLCH);
        cv::Mat lab2lch(const cv::Mat& imgLAB);
        cv::Mat rgb2xyz(const cv::Mat& img);
        cv::Mat xyz2rgb(const cv::Mat& img);
        cv::Mat xyz2lab(const cv::Mat& img);
        cv::Mat lab2xyz(const cv::Mat& img);


};



