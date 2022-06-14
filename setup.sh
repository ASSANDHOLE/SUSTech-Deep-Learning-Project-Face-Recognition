echo "This script will create a resources directory (./resources) and populate it with the following files(dirs):"
echo "  - ./config.yaml"
echo "  - ./resources/weights/net_weights*"
echo "  - ./resources/known_faces"
echo "  - ./resources/videos"
echo "And also clone the third-party mentioned in the third_party_libs/readme.md file"

echo "The config file (./config.yaml) is the template for running the project on YOUR COMPUTER'S WEBCAM (VideoCapture(0))"
echo "The (./resources/known_faces) is the directory where you put your known faces (as described in the readme)"

cd third_party_libs || exit
if [ -d dlib ]; then
  echo "dlib directory found, skipping..."
else
  git clone https://github.com/davisking/dlib.git
fi
if [ -d yaml-cpp ]; then
  echo "yaml-cpp directory found, skipping..."
else
  git clone https://github.com/jbeder/yaml-cpp.git
fi
cd ..

[ -d resources ] || mkdir resources
[ -d resources/weights ] || mkdir resources/weights
[ -d resources/known_faces ] || mkdir resources/known_faces
[ -d resources/videos ] || mkdir resources/videos
if [ -f resources/weights/dlib_face_recognition_resnet_model_v1.dat ]; then
  echo "dlib_face_recognition_resnet_model_v1.dat found, skipping..."
else
  wget -O resources/weights/dlib_face_recognition_resnet_model_v1.dat.bz2 http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
  bunzip2 resources/weights/dlib_face_recognition_resnet_model_v1.dat.bz2
fi

if [ -f resources/weights/mmod_human_face_detector.dat.bz2 ]; then
  echo "mmod_human_face_detector.dat found, skipping..."
else
  wget -O resources/weights/mmod_human_face_detector.dat.bz2 http://dlib.net/files/mmod_human_face_detector.dat.bz2
  bunzip2 resources/weights/mmod_human_face_detector.dat.bz2
fi

if [ -f resources/weights/shape_predictor_5_face_landmarks.dat ]; then
  echo "shape_predictor_5_face_landmarks.dat found, skipping..."
else
  wget -O resources/weights/shape_predictor_5_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
  bunzip2 resources/weights/shape_predictor_5_face_landmarks.dat.bz2
fi

current_dir=$(pwd)

if [ -f ./config.yaml ]; then
  echo "config.yaml found, skipping..."
else
  {
    echo "weight:";
    echo "  face_recognition_model_path: $current_dir/resources/weights/dlib_face_recognition_resnet_model_v1.dat";
    echo "  mmod_human_face_detector_model_path: $current_dir/resources/weights/mmod_human_face_detector.dat";
    echo "  shape_predictor_model_path: $current_dir/resources/weights/shape_predictor_5_face_landmarks.dat";
    echo "face_recognition_threshold: 0.35";
    echo "use_video: false";
    echo "video_path: $current_dir/resources/videos/YOUR_VIDEO.MP4";
    echo "known_faces_path: $current_dir/resources/known_faces";
    echo "on_jetson: false";
  } >> ./config.yaml
fi
