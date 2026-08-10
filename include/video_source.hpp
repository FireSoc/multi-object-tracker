// Classes for starting a video
#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

struct VideoSource {
    explicit VideoSource(int camera_index = 0);
    bool is_open() const;
    bool next(cv::Mat& frame);
    cv::VideoCapture capture_;

};

