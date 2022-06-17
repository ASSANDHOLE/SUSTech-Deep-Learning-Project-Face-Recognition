//
// Created by anguangyan on 5/20/22.
//

#include "dlib/dnn.h"
#include "dlib/clustering.h"
#include "dlib/string.h"

#include <opencv2/opencv.hpp>

#include "recognition_utils.h"
#include "config_utils.h"
#include <video_utils.h>

#ifdef CONFIG_PATH
const char *kConfigPath = CONFIG_PATH;
#else
const char *kConfigPath = "./config.yaml";
#endif


void RecognizeFace(cv::VideoCapture &cap, const Config &config) {
    const cv::Scalar UNKNOW_COLOR(0, 0, 255);  // Red (BGR)
    dlib::shape_predictor sp;
    dlib::deserialize(config.shape_predictor_model_path) >> sp;
    anet_type net;
    dlib::deserialize(config.face_recognition_model_path) >> net;
    net_type detector;
    dlib::deserialize(config.mmod_human_face_detector_model_path) >> detector;

    auto known_face_files = ListDirectory(config.known_face_path);
    auto known_face_names = GetFileName(known_face_files);
    auto known_faces = LoadImages(known_face_files);
    auto colour_list = GetColours(known_face_names.size());
    std::vector<face_descriptor_t> known_face_descriptors = net(known_faces);
    // auto cap = CreateVideoCapture(config.use_video, config.on_jetson, config.video_path);
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

            std::vector<std::tuple<dlib::rectangle, std::string, cv::Scalar>> result;
            for (size_t i = 0; i < face_descriptors.size(); ++i) {
                int which = -1;
                int unknow = -1;
                double dist;
                double best_dist = 1e6;
                double best_unknow_dist = 1e6;
                for (int j = 0; j < known_face_descriptors.size(); ++j) {
                    if ((dist = length(face_descriptors[i] - known_face_descriptors[j])) < config.face_recognition_threshold) {
                        // dist = length(face_descriptors[i] - known_face_descriptors[j]);
                        printf("%s: %.2f\n", known_face_names[j].c_str(), dist);
                        if (dist < best_dist) {
                            best_dist = dist;
                            which = j;
                        }
                    }
                    else {
                        
                        if (dist < best_unknow_dist) {
                            best_unknow_dist = dist;
                            unknow = j;
                        }
                    }
                    // printf("dist: %.2f\n",  dist);
                }
                if (which >= 0) {
                    result.emplace_back(face_rects[i], known_face_names[which], colour_list[which]);
                }
                else if (unknow >= 0){
                    printf("find unknow face\n");
                    result.emplace_back(face_rects[i], "unknow", UNKNOW_COLOR);
                }
            }
            DrawRectangleWithName(frame, result);
        }
        cv::imshow("result", frame);
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }
}

int main(int argc, char **argv) {
    try {
        auto config = ReadConfig(kConfigPath);
        auto cap = CreateVideoCapture(config.use_video, config.on_jetson, config.video_path);
        RecognizeFace(cap, config);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}
