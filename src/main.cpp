//
// Created by anguangyan on 5/20/22.
//

#include "dlib/dnn.h"
#include "dlib/clustering.h"
#include "dlib/string.h"

#include <opencv2/opencv.hpp>

#include "recognition_utils.h"
#include "config_utils.h"

#ifdef CONFIG_PATH
const char *kConfigPath = CONFIG_PATH;
#else
const char *kConfigPath = "./config.yaml";
#endif

std::string JetsonNanoGstreamerPipeline(int capture_width, int capture_height,
                                        int display_width, int display_height,
                                        int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}


int main(int argc, char **argv) try {
    auto config = ReadConfig(kConfigPath);
    const cv::Scalar COLOR(0, 0, 255);  // Red (BGR)
    dlib::shape_predictor sp;
    dlib::deserialize(config.shape_predictor_model_path) >> sp;
    anet_type net;
    dlib::deserialize(config.face_recognition_model_path) >> net;
    net_type detector;
    dlib::deserialize(config.mmod_human_face_detector_model_path) >> detector;

    auto known_face_files = ListDirectory(config.known_face_path);
    auto known_face_names = GetFileName(known_face_files);
    auto known_faces = LoadImages(known_face_files);
    std::vector<face_descriptor_t> known_face_descriptors = net(known_faces);
    cv::VideoCapture cap;
    if (config.use_video) {
        cap = cv::VideoCapture(config.video_path);
    } else if (config.on_jetson) {
        int _capture_width = 640;
        int _capture_height = 360;
        int _display_width = 640;
        int _display_height = 360;
        int _framerate = 10;
        int _flip_method = 0;
        std::string _pipeline = JetsonNanoGstreamerPipeline(_capture_width,
                                                            _capture_height,
                                                            _display_width,
                                                            _display_height,
                                                            _framerate,
                                                            _flip_method);
        std::cout << "Using pipeline: \n\t" << _pipeline << "\n";
        cap = cv::VideoCapture(_pipeline, cv::CAP_GSTREAMER);
    } else {
        cap = cv::VideoCapture(0);
    }
    if (!cap.isOpened()) {
        throw std::runtime_error("can't open camera");
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 0);
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            if (config.use_video) {
                break;
            } else {
                std::cout << "empty frame" << std::endl;
                continue;
            }
        }
        if (config.use_video) {
            cv::flip(frame, frame, 1);
        }
        auto img = FromCvMat(frame);
        std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
        std::vector<dlib::rectangle> face_rects;
        for (auto&& face: detector(img)) {
            auto shape = sp(img, face);
            dlib::matrix<dlib::rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(std::move(face_chip));
            face_rects.push_back(face);
        }

        if (faces.empty()) {
            std::cout << "no face detected" << std::endl;
        } else {
            std::vector<face_descriptor_t> face_descriptors = net(faces);

            std::vector<std::pair<dlib::rectangle, std::string>> result;
            for (size_t i = 0; i < face_descriptors.size(); ++i) {
                int which = -1;
                double dist;
                double best_dist = 1e6;
                for (int j = 0; j < known_face_descriptors.size(); ++j) {
                    if ((dist = length(face_descriptors[i] - known_face_descriptors[j])) < config.face_recognition_threshold) {
                        // dist = length(face_descriptors[i] - known_face_descriptors[j]);
                        printf("%s: %.2f\n", known_face_names[j].c_str(), dist);
                        if (dist < best_dist) {
                            best_dist = dist;
                            which = j;
                        }
                    }
                }
                if (which >= 0) {
                    result.emplace_back(face_rects[i], known_face_names[which]);
                }
            }
            DrawRectangleWithName(frame, result, COLOR);
        }
        cv::imshow("result", frame);
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }
} catch (std::exception &e) {
    std::cout << e.what() << std::endl;
}
