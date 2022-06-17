//
// Created by anguangyan on 6/15/22.
//

#ifndef DEEP_LEARNING_PROJECT_VIDEO_UTILS_H
#define DEEP_LEARNING_PROJECT_VIDEO_UTILS_H

#include <opencv2/opencv.hpp>
#include <atomic>

#include <thread>

inline std::string JetsonNanoGstreamerPipeline(int capture_width, int capture_height,
                                        int display_width, int display_height,
                                        int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

cv::VideoCapture CreateVideoCapture(bool use_video, bool on_jetson, const std::string &video_path="");

class NoDelayCameraCapture {
public:
    inline explicit NoDelayCameraCapture(cv::VideoCapture &&cap) : cap_(cap) {
        if (!cap.isOpened()) {
            throw std::runtime_error("The video capture is not opened.");
        }
        start();
    }

    inline bool start() {
        running_ = true;
        worker_ = std::thread([this]() {run();});
        return true;
    }

    inline bool stop() {
        running_ = false;
        worker_.join();
        return true;
    }

    cv::Mat get();

private:
    void run();
    cv::Mat tmp_;
    cv::Mat image_;
    cv::VideoCapture cap_;
    // std::atomic<bool> is_using_{false};
    std::atomic<bool> has_new_{false};
    std::mutex mutex_{};
    std::thread worker_;
    bool running_{true};
};

class VideoCaptureWrapper {
public:
    inline explicit VideoCaptureWrapper(cv::VideoCapture &&cap) : cap_(cap) {
        if (!cap.isOpened()) {
            throw std::runtime_error("The video capture is not opened.");
        }
    }

    inline cv::Mat get() {
        cv::Mat res;
        cap_ >> res;
        return res;
    }

private:
    cv::VideoCapture cap_;
};

#endif //DEEP_LEARNING_PROJECT_VIDEO_UTILS_H
