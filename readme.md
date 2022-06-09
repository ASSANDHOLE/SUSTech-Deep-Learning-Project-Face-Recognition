# Face Recognition Project

### How to run
Before you start, make sure to download the required libraries. Follow the instructions in [third_party_libs/readme.md](third_party_libs/readme.md).

Also, I'm not sure if the list dir method is going to work on Windows. You can modify it in [src/utils.cpp](src/utils.cpp).

Create a config file `config.yaml` at project source directory (can be configured in the `CMakeLists.txt` file).

Template config file (Please use full path):

```yaml
# this is the template for running the project
# on YOUR COMPUTER'S WEBCAM (VideoCapture(0))
weight:
  # could be found at http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2, unzip before use
  face_recognition_model_path: /path/to/face_recognition_model
  # could be found at http://dlib.net/files/mmod_human_face_detector.dat.bz2, unzip before use
  mmod_human_face_detector_model_path: /path/to/mmod_human_face_detector_model
  # could be found at http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2, unzip before use
  shape_predictor_model_path: /path/to/shape_predictor_model

# should between 0 and 1, the smaller, the more strict
face_recognition_threshold: 0.35
# use a video file instead of webcam as video input
use_video: false
# if use_video is true, you should set the following to the video path
video_path: /path/to/video
# known_face_path is the directory include all the known faces (not arbitrary image with face), i.e. ["name1.png", "name2.png", ...]
# you could generate face by the utility python script or target "create_new_face"
known_face_path: /path/to/known_face_path
# if on_jetson is true and use_video is false, the Gstreamer will be used to capture video on Jetson Nano
on_jetson: false
```

Then, run the following command to build the project:

```shell
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Finally, run target `face_rec_dnn` to run face recognition.

Or, run target `create_new_face` to extract face from an image.

