//
// Created by anguangyan on 5/20/22.
//

#ifndef EXAMPLES_RECOGNITION_UTILS_H
#define EXAMPLES_RECOGNITION_UTILS_H

#include <string>
#include <vector>

#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/matrix/matrix.h>
#include <dlib/pixel.h>

#include <opencv2/opencv.hpp>

// -------------------------------- Define Net Structure ----------------------------------

template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template<int N, template<typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template<int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template<int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template<typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template<typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template<typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template<typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template<typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = dlib::loss_metric<
        dlib::fc_no_bias<128, dlib::avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<dlib::max_pool<
                3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2, dlib::input_rgb_image_sized<150>
                >>>>>>>>>>>>;
// ----------------------------------------------------------------------------------------

template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = dlib::con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler = dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = dlib::relu<dlib::affine<con5<45,SUBNET>>>;

using net_type = dlib::loss_mmod<dlib::con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

using image_t = dlib::matrix<dlib::rgb_pixel>;
using file_names_t = std::vector<std::string>;
using face_descriptor_t = dlib::matrix<float, 0, 1>;

std::vector<image_t> LoadImages(const file_names_t &paths);

file_names_t ListDirectory(const std::string &path);

file_names_t GetNewFaces(const file_names_t &old_faces, const file_names_t &new_faces);

image_t FromCvMat(const cv::Mat &mat);

cv::Mat ToCvMat(image_t &img);

void DrawRectangleWithName(cv::Mat &img, const std::vector<std::pair<dlib::rectangle, std::string>> &rect, const cv::Scalar &color);

std::vector<std::string> GetFileName(const std::vector<std::string> &paths);


#endif //EXAMPLES_RECOGNITION_UTILS_H
