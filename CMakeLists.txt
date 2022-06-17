cmake_minimum_required(VERSION 3.4)
project(DeepLearningProject)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(./third_party_libs/dlib dlib_build)
add_subdirectory(./third_party_libs/yaml-cpp yaml_cpp_build)

add_definitions(-DCONFIG_PATH="${PROJECT_SOURCE_DIR}/config.yaml")

set(EXEC_NAME face_rec_dnn)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)

add_executable(${EXEC_NAME} src/main.cpp src/utils.cpp)
target_include_directories(${EXEC_NAME} PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${EXEC_NAME} yaml-cpp::yaml-cpp dlib::dlib ${OpenCV_LIBS})

add_executable(create_new_face src/create_new_face.cpp src/utils.cpp)
target_include_directories(create_new_face PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(create_new_face yaml-cpp::yaml-cpp dlib::dlib ${OpenCV_LIBS})
