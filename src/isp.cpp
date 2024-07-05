#include <iostream>
#include <fstream>
#include <memory>
#include "isp_pipeline.hpp"
#include "utils.hpp"
#include "confini.hpp"
#include <chrono>

int main(int argc, char* argv[]) {
    // parse ini file
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config-file>" << std::endl;
        return 1;
    }

    ConfigIni config(argv[1]);
    if (!config.load()){
        std::cerr << "Failed to load the configuration file." << std::endl;
    }
    
    //const std::string image_name = "DSC_1339_768x512_rggb";            // image content: Rose rggb
    // image_name = "DSC_1339_768x512_gbrg"            // image content: Rose gbrg
    // image_name = "DSC_1339_768x512_grbg"            // image content: Rose grbg
    // image_name = "DSC_1339_768x512_bggr"            // image content: Rose bggr
    const std::string image_name = "DSC_1320_2048x2048_rggb";        // image content: Potrait
    // image_name = "DSC_1372_6032x4032_rggb"        // image content: Downtown San Jose
    // image_name = "DSC_1372_12096x6032_rgb_out_demosaic" // image content: Downtown San Jose after demosaic


    auto obj = std::make_unique<isp_pipeline>(image_name, config);
    
    // load raw data
    obj->load_raw_data();
    
    obj->create_cv_mat_data();
    
    imsave(obj->get_cv_mat(),"./images/DSC_1320_raw.png");
    obj->black_level_correction();

    imsave(obj->get_cv_mat(),"./images/DSC_1320_black_level_correction.png");

    int neighborhood_size =3;    
    //auto start = std::chrono::high_resolution_clock::now();
    obj->bad_pixel_correction_cuda(neighborhood_size);

    imsave(obj->get_cv_mat(),"./images/DSC_1320_bad_pixel_correction.png");
    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> duration = end - start;
    //std::cout << "Duration: " << duration.count() << " seconds" << std::endl;
    obj->do_channel_gain_white_balance();
    
    imsave(obj->get_cv_mat(),"./images/DSC_1320_white_balance.png");
    
    obj->do_demosaic();

    imsave(obj->get_cv_mat(),"./images/DSC_1320_demosaic.png");

    float beta = 0.60 * 65535;
    obj->post_process_local_color_ratio(beta);
    
    imsave(obj->get_cv_mat(),"./images/DSC_1320_local_color_ratio.png");
    obj->apply_cmatrix();
    
    imsave(obj->get_cv_mat(),"./images/DSC_1320_apply_color_matrix.png");
    obj->do_gamma();
    
    imsave(obj->get_cv_mat(),"./images/DSC_1320_gamma.png");

    float nsr_threshold = 90.0;
    float cr_threshold = 6425.0*0.5;
    
    obj->purple_fringe_removal(nsr_threshold, cr_threshold);
    
    imsave(obj->get_cv_mat(),"./images/DSC_1320_purple_fringe_removal.png");
    
    obj->do_tone_mapping();
   
    imsave(obj->get_cv_mat(),"./images/DSC_1320_tone_mapping.png");


    //auto start = std::chrono::high_resolution_clock::now();
    obj->equalizeHistColor_opencv();
    //auto end = std::chrono::high_resolution_clock::now();
    //duration = end - start;

    imsave(obj->get_cv_mat(), "./images/DSC_1320_equalizeHistColor.png");
    //std::cout << "Duration: " << duration.count() << " seconds" << std::endl;
    //cv::Mat image =  create_mat_from_raw_data(image_name, 768, 512);
    // cv::Mat image = cv::imread("/home/fuli2/sosf.png");
    //if (image.empty()) {
    //   std::cerr << "Failed to load image." << std::endl;
    //    return -1;
    //}
    obj->do_sharpening();
    imsave(obj->get_cv_mat(),"./images/DSC_1320_sharpening.png");
    //cv::imshow("Loaded Image", obj->get_cv_mat());
    //cv::waitKey(0);

    return 0;
}

