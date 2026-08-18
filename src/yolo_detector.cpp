#include "yolo_detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

constexpr int kBoxCoordinateCount = 4;
constexpr int kMinimumFeatureCount = kBoxCoordinateCount + 1;
constexpr double kPixelScale = 1.0 / 255.0;

cv::Mat candidate_rows(const cv::Mat& raw_output) {
    constexpr int kExpectedDimensions = 3;
    constexpr int kBatchSize = 1;

    if (raw_output.dims != kExpectedDimensions || raw_output.size[0] != kBatchSize ||
        raw_output.channels() != 1) {
        throw std::runtime_error(
            "YOLO output must have shape [1, features, candidates] or "
            "[1, candidates, features]");
    }

    cv::Mat output;
    raw_output.convertTo(output, CV_32F);
    if (!output.isContinuous()) {
        output = output.clone();
    }

    const int first_size = output.size[1];
    const int second_size = output.size[2];
    const bool features_are_first =
        first_size >= kMinimumFeatureCount &&
        (first_size <= second_size || second_size < kMinimumFeatureCount);

    const int feature_count = features_are_first ? first_size : second_size;
    const int candidate_count = features_are_first ? second_size : first_size;
    if (feature_count < kMinimumFeatureCount || candidate_count <= 0) {
        throw std::runtime_error("YOLO output has no class scores or candidates");
    }

    cv::Mat matrix(feature_count, candidate_count, CV_32F, output.ptr<float>());
    if (features_are_first) {
        return matrix.t();
    }

    return cv::Mat(candidate_count, feature_count, CV_32F, output.ptr<float>()).clone();
}

}  // namespace

YoloDetector::YoloDetector(
    const std::string& model_path,
    float confidence_threshold,
    float nms_threshold)
    : confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold) {
    if (model_path.empty()) {
        throw std::invalid_argument("YOLO model path must not be empty");
    }
    if (confidence_threshold < 0.0F || confidence_threshold > 1.0F) {
        throw std::invalid_argument("YOLO confidence threshold must be in [0, 1]");
    }
    if (nms_threshold < 0.0F || nms_threshold > 1.0F) {
        throw std::invalid_argument("YOLO NMS threshold must be in [0, 1]");
    }

    try {
        network_ = cv::dnn::readNetFromONNX(model_path);
    } catch (const cv::Exception& exception) {
        throw std::runtime_error(
            "Failed to load YOLO ONNX model '" + model_path + "': " + exception.what());
    }

    if (network_.empty()) {
        throw std::runtime_error("Failed to load YOLO ONNX model '" + model_path + "'");
    }
}

std::vector<DetectedObject> YoloDetector::detect(const cv::Mat& frame) {
    if (frame.empty()) {
        throw std::invalid_argument("Cannot run YOLO detection on an empty frame");
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(
        frame,
        blob,
        kPixelScale,
        cv::Size(kInputSize, kInputSize),
        cv::Scalar(),
        true,
        false);

    network_.setInput(blob);
    const cv::Mat output = network_.forward();
    const cv::Mat candidates = candidate_rows(output);

    const float horizontal_scale = static_cast<float>(frame.cols) / kInputSize;
    const float vertical_scale = static_cast<float>(frame.rows) / kInputSize;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    boxes.reserve(candidates.rows);
    confidences.reserve(candidates.rows);

    for (int row_index = 0; row_index < candidates.rows; ++row_index) {
        const float* row = candidates.ptr<float>(row_index);
        const float* class_scores = row + kBoxCoordinateCount;
        const float* class_scores_end = row + candidates.cols;
        const float confidence = *std::max_element(class_scores, class_scores_end);
        if (!std::isfinite(confidence) || confidence < confidence_threshold_) {
            continue;
        }

        const float center_x = row[0];
        const float center_y = row[1];
        const float width = row[2];
        const float height = row[3];
        if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
            !std::isfinite(width) || !std::isfinite(height) || width <= 0.0F ||
            height <= 0.0F) {
            continue;
        }

        const float left_coordinate = std::clamp(
            (center_x - width / 2.0F) * horizontal_scale, 0.0F,
            static_cast<float>(frame.cols));
        const float top_coordinate = std::clamp(
            (center_y - height / 2.0F) * vertical_scale, 0.0F,
            static_cast<float>(frame.rows));
        const float right_coordinate = std::clamp(
            (center_x + width / 2.0F) * horizontal_scale, 0.0F,
            static_cast<float>(frame.cols));
        const float bottom_coordinate = std::clamp(
            (center_y + height / 2.0F) * vertical_scale, 0.0F,
            static_cast<float>(frame.rows));
        const int left = static_cast<int>(std::floor(left_coordinate));
        const int top = static_cast<int>(std::floor(top_coordinate));
        const int right = static_cast<int>(std::ceil(right_coordinate));
        const int bottom = static_cast<int>(std::ceil(bottom_coordinate));
        if (right <= left || bottom <= top) {
            continue;
        }

        boxes.emplace_back(left, top, right - left, bottom - top);
        confidences.push_back(confidence);
    }

    std::vector<int> kept_indices;
    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        confidence_threshold_,
        nms_threshold_,
        kept_indices);

    std::vector<DetectedObject> detections;
    detections.reserve(kept_indices.size());
    for (const int index : kept_indices) {
        if (index < 0 || index >= static_cast<int>(boxes.size())) {
            throw std::runtime_error("YOLO NMS returned an invalid candidate index");
        }
        detections.push_back({boxes[index], confidences[index]});
    }

    return detections;
}
