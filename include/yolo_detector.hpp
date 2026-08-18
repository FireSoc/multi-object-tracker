#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "detection.hpp"

class YoloDetector {
   public:
    static constexpr int kInputSize = 640;
    static constexpr float kDefaultConfidenceThreshold = 0.25F;
    static constexpr float kDefaultNmsThreshold = 0.45F;

    explicit YoloDetector(
        const std::string& model_path,
        float confidence_threshold = kDefaultConfidenceThreshold,
        float nms_threshold = kDefaultNmsThreshold);

    std::vector<DetectedObject> detect(const cv::Mat& frame);

   private:
    cv::dnn::Net network_;
    float confidence_threshold_;
    float nms_threshold_;
};
