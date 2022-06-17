//
// Created by anguangyan on 5/20/22.
//

#include <recognition_utils.h>
#include <config_utils.h>
#include <video_utils.h>

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include <dirent.h>
#include <algorithm>

#include <dlib/opencv.h>

#include <yaml-cpp/yaml.h>

// recognition_utils.h
std::vector<image_t> LoadImages(const file_names_t &paths) {
    std::vector<image_t> images;
    for (auto &path : paths) {
        image_t image;
        dlib::load_image(image, path);
        images.push_back(std::move(image));
    }
    return images;
}


file_names_t ListDirectory(const std::string &path) {
    file_names_t res;
    DIR *d;
    struct dirent *dir;
    d = opendir(path.c_str());
    if (d) {
        while ((dir = readdir(d)) != nullptr) {
            if (dir->d_type == DT_REG) {
                res.emplace_back(path + dir->d_name);
            }
        }
        closedir(d);
    }
    return res;
}

file_names_t GetNewFaces(const file_names_t &old_faces, const file_names_t &new_faces) {
    file_names_t res;
    for (const auto &face : new_faces) {
        if (std::find(old_faces.begin(), old_faces.end(), face) == old_faces.end()) {
            res.emplace_back(face);
        }
    }
    return res;
}

image_t FromCvMat(const cv::Mat &mat) {
    image_t image;
    dlib::assign_image(image, dlib::cv_image<dlib::bgr_pixel>(mat));
    return image;
}

cv::Mat ToCvMat(image_t &img) {
    return dlib::toMat(img);
}

inline cv::Rect DlibRectToCvRect(const dlib::rectangle &r) {
    return {cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1)};
}

void DrawRectangleWithName(cv::Mat &img, const std::vector<std::tuple<dlib::rectangle, std::string, cv::Scalar>> &rect){
    for (const auto & i : rect) {
        auto cv_rect = DlibRectToCvRect(std::get<0>(i));
        if (std::get<1>(i).compare("unknow") == 0){
            cv::rectangle(img, cv_rect, std::get<2>(i), 2);
            cv::putText(img, std::get<1>(i), cv_rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 2, std::get<2>(i), 2);
        }
        else{
            cv::rectangle(img, cv_rect, std::get<2>(i), 2);
            cv::putText(img, std::get<1>(i), cv_rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 2, std::get<2>(i), 2);
        }
    }
}

std::vector<std::string> GetFileName(const std::vector<std::string> &paths) {
    std::vector<std::string> res;
    for (const auto & path : paths) {
        auto pos = path.find_last_of('/');
        auto pos2 = path.find_last_of('.');
        if (pos == std::string::npos) {
            res.push_back(path.substr(0, pos2));
        } else if (pos2 == std::string::npos) {
            res.push_back(path.substr(pos + 1));
        } else {
            res.push_back(path.substr(pos + 1, pos2 - pos - 1));
        }
    }
    return res;
}

std::vector<cv::Scalar> GetColours(const size_t len){
    std::vector<cv::Scalar> res;
    for (size_t i = 0; i < len; i++) {
        res.push_back(cv::Scalar(rand()&255, rand()&255, rand()&255));
    }
    return res;
}
// end recognition_utils.h

class NodeWrapper {
public:
    explicit NodeWrapper(YAML::Node &&node) : node_(node) {}
    NodeWrapper operator[](const std::string &key) const {
        try {
            auto new_node = node_[key];
            auto new_wrapper = NodeWrapper(std::move(new_node));
            new_wrapper.parents_ = std::vector<std::string>(parents_);
            if (!key_.empty()) {
                new_wrapper.parents_.push_back(key_);
            }
            new_wrapper.key_ = key;
            return new_wrapper;
        } catch (std::exception &e) {
            std::stringstream ss;
            ss << "Key \"";
            for (auto &parent : parents_) {
                ss << parent << ".";
            }
            if (!key_.empty()) {
                ss << key_ << ".";
            }
            ss << key;
            ss << "\" Not found in node.\n";
            throw std::runtime_error(ss.str());
        }
    }
    template <typename T>
    T as() const {
        try {
            return node_.as<T>();
        } catch (std::exception &e) {
            std::stringstream ss;
            ss << "Key \"";
            for (auto &parent : parents_) {
                ss << parent << ".";
            }
            if (!key_.empty()) {
                ss << key_;
            } else {
                ss.seekp(-1, std::stringstream::cur);
            }
            ss << "\" not exist OR failed to be interpreted as " << typeid(T).name() << ".\n";
            throw std::runtime_error(ss.str());
        }
    }

    template <typename T, typename S>
    T as(const S& fallback) const {
        return node_.as<T>(fallback);
    }

private:
    const YAML::Node node_;
    std::string key_;
    std::vector<std::string> parents_;
};

// config_utils.h
Config ReadConfig(const std::string &config_path) {
    YAML::Node raw_config = YAML::LoadFile(config_path);
    NodeWrapper config(std::move(raw_config));
    // YAML::Node config = YAML::LoadFile(config_path);
    auto weight_node = config["weight"];
    auto face_recognition_model_path = weight_node["face_recognition_model_path"].as<std::string>();
    auto mmod_human_face_detector_model_path = weight_node["mmod_human_face_detector_model_path"].as<std::string>();
    auto shape_predictor_model_path = weight_node["shape_predictor_model_path"].as<std::string>();

    auto face_recognition_threshold = config["face_recognition_threshold"].as<double>();
    auto use_video = config["use_video"].as<bool>();
    auto video_path = config["video_path"].as<std::string>();
    auto known_face_path = config["known_face_path"].as<std::string>();
    auto on_jetson = config["on_jetson"].as<bool>();
    if (known_face_path[known_face_path.size() - 1] != '/') {
        known_face_path += '/';
    }
    return {
        face_recognition_model_path,
        mmod_human_face_detector_model_path,
        shape_predictor_model_path,
        face_recognition_threshold,
        use_video,
        video_path,
        known_face_path,
        on_jetson
    };
}

bool WriteConfig(const Config &config, const std::string &config_path) {
    YAML::Node config_node;
    YAML::Node weight_node;
    weight_node["face_recognition_model_path"] = config.face_recognition_model_path;
    weight_node["mmod_human_face_detector_model_path"] = config.mmod_human_face_detector_model_path;
    weight_node["shape_predictor_model_path"] = config.shape_predictor_model_path;

    config_node["weight"] = weight_node;
    config_node["face_recognition_threshold"] = config.face_recognition_threshold;
    config_node["use_video"] = config.use_video;
    config_node["video_path"] = config.video_path;
    config_node["known_face_path"] = config.known_face_path;
    config_node["on_jetson"] = config.on_jetson;
    std::ofstream fout(config_path);
    fout << config_node;
    fout.close();
    return true;
}
// end config_utils.h

// video_utils.h
cv::VideoCapture CreateVideoCapture(bool use_video, bool on_jetson, const std::string &video_path) {
    if (use_video && video_path.empty()) {
        throw std::runtime_error("Video path is not specified.");
    }
    cv::VideoCapture cap;
    if (use_video) {
        cap = cv::VideoCapture(video_path);
    } else if (on_jetson) {
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
        throw std::runtime_error("can't open VideoCapture}");
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 0);
    return cap;
}
// end video_utils.h
