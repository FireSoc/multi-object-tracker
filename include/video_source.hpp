// Classes for starting a video
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>

struct VideoSource {
    explicit VideoSource(int camera_index = 0);
    explicit VideoSource(const std::string& video_path);

    bool is_open() const;
    bool next(cv::Mat& frame);

    cv::VideoCapture capture_;
};
