//
// Created by anguangyan on 6/9/22.
//

#ifndef EXAMPLES_CONFIG_UTILS_H
#define EXAMPLES_CONFIG_UTILS_H

struct Config {
    const std::string face_recognition_model_path;
    const std::string mmod_human_face_detector_model_path;
    const std::string shape_predictor_model_path;
    const double face_recognition_threshold;
    const bool use_video;
    const std::string video_path;
    const std::string known_face_path;
    const bool on_jetson;
};

Config ReadConfig(const std::string &config_path);

bool WriteConfig(const Config &config, const std::string &config_path);


#endif //EXAMPLES_CONFIG_UTILS_H
