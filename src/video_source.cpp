// src/video_source.cpp
#include "video_source.hpp"

VideoSource::VideoSource(int camera_index) : capture_(camera_index) {}

VideoSource::VideoSource(const std::string& video_path) : capture_(video_path) {}

bool VideoSource::is_open() const { return capture_.isOpened(); }

bool VideoSource::next(cv::Mat& frame) { return capture_.read(frame); }
