// Header for motion_detector.cpp that defines MotionDetector
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#include "detection.hpp"

struct MotionDetector {

    cv::Ptr<cv::BackgroundSubtractor> subtractor = cv::createBackgroundSubtractorMOG2();
    double min_area = 500.0;
    std::vector<DetectedObject> detect(const cv::Mat& frame);
};

