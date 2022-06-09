//
// Created by anguangyan on 5/20/22.
//

#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <config_utils.h>

#include <vector>

#ifdef CONFIG_PATH
const char *kConfigPath = CONFIG_PATH;
#else
const char *kConfigPath = "./config.yaml";
#endif

auto getDetectorAndShapePredictor() {
    using namespace dlib;
    using namespace std;

    static bool loaded = false;

    auto config = ReadConfig(kConfigPath);

    const static frontal_face_detector detector = get_frontal_face_detector();
    static shape_predictor sp;
    if (!loaded) {
        deserialize(config.shape_predictor_model_path) >> sp;
        loaded = true;
    }
    return make_pair(detector, sp);
}

extern "C" int CreateNewFace(const char *img_path, const char *save_path) {
    using namespace dlib;
    std::vector<matrix<rgb_pixel>> faces;
    auto dec = getDetectorAndShapePredictor();
    auto detector = dec.first;
    auto sp = dec.second;
    matrix<rgb_pixel> img;
    load_image(img, img_path);
    auto res = detector(img);
    if (res.empty()) {
        return -1;
    }
    auto face = res[0];
    auto shape = sp(img, face);
    matrix<rgb_pixel> face_chip;
    dlib::extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
    save_png(face_chip, save_path);
    return 0;
}

int main(int argc, char **argv) {
    using namespace dlib;
    using namespace std;

    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <img_path> <save_path>" << endl;
        return 1;
    }
    return CreateNewFace(argv[1], argv[2]);
}
