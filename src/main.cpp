//
// Created by anguangyan on 5/20/22.
//

#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <cstring>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <recognition_utils.h>
#include <config_utils.h>
#include <video_utils.h>

#ifdef CONFIG_PATH
const char *kConfigPath = CONFIG_PATH;
#else
const char *kConfigPath = "./config.yaml";
#endif

void DrawFps(cv::Mat &frame, double fps) {
    fps = fps > 1000.0 ? 0.0 : fps;
    char str[20];
    sprintf(str, "FPS: %.2f", fps);
    cv::putText(frame, str, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 3);
}

template <typename CaptureWrapper>
void RecognizeFace(CaptureWrapper &cap, const Config &config) {
    const cv::Scalar UNKNOWN_COLOR(0, 0, 255);  // Red (BGR)
    dlib::shape_predictor sp;
    dlib::deserialize(config.shape_predictor_model_path) >> sp;
    anet_type net;
    dlib::deserialize(config.face_recognition_model_path) >> net;
    net_type detector;
    dlib::deserialize(config.mmod_human_face_detector_model_path) >> detector;

    auto known_face_files = ListDirectory(config.known_face_path);
    auto known_face_names = GetFileName(known_face_files);
    auto known_faces = LoadImages(known_face_files);
    auto colour_list = GetColors(known_face_names.size());
    std::vector<face_descriptor_t> known_face_descriptors = net(known_faces);
    // auto cap = CreateVideoCapture(config.use_video, config.on_jetson, config.video_path);
    auto last_frame_time = std::chrono::steady_clock::now();
    auto this_frame_time = std::chrono::steady_clock::now();
    long duration;

    while (true) {
        cv::Mat frame = cap.get();
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
            continue;
        } else {
            std::vector<face_descriptor_t> face_descriptors = net(faces);

            std::vector<std::tuple<dlib::rectangle, std::string, cv::Scalar>> result;
            for (size_t i = 0; i < face_descriptors.size(); ++i) {
                int which = -1;
                int unknown = -1;
                double dist;
                double best_dist = 1e6;
                double best_unknown_dist = 1e6;
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
                        
                        if (dist < best_unknown_dist) {
                            best_unknown_dist = dist;
                            unknown = j;
                        }
                    }
                    // printf("dist: %.2f\n",  dist);
                }
                if (which >= 0) {
                    result.emplace_back(face_rects[i], known_face_names[which], colour_list[which]);
                }
                else if (unknown >= 0){
                    printf("find unknown face\n");
                    result.emplace_back(face_rects[i], "unknown", UNKNOWN_COLOR);
                }
            }
            DrawRectangleWithName(frame, result);
        }
        this_frame_time = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(this_frame_time - last_frame_time).count();
        last_frame_time = this_frame_time;
        double fps = 1.0 / (duration * 1e-9);
        DrawFps(frame, fps);
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
        if (config.use_video) {
            VideoCaptureWrapper res{std::move(cap)};
            RecognizeFace(res, config);
        } else {
            NoDelayCameraCapture res{std::move(cap)};
            RecognizeFace(res, config);
        }
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}
