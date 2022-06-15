//
// Created by anguangyan on 6/15/22.
//

#ifndef DEEP_LEARNING_PROJECT_VIDEO_UTILS_H
#define DEEP_LEARNING_PROJECT_VIDEO_UTILS_H

#include <opencv2/opencv.hpp>

inline std::string JetsonNanoGstreamerPipeline(int capture_width, int capture_height,
                                        int display_width, int display_height,
                                        int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

cv::VideoCapture CreateVideoCapture(bool use_video, bool on_jetson, const std::string &video_path="");

#endif //DEEP_LEARNING_PROJECT_VIDEO_UTILS_H
