// Cpp file defining a function detect

#include <vector>

#include <opencv2/geometry.hpp> 
#include <opencv2/imgproc.hpp>

#include "motion_detector.hpp"

std::vector<DetectedObject> MotionDetector::detect(const cv::Mat& frame) {
    std::vector<DetectedObject> objects;

    // 1. Frame -> foreground mask (0 = background, 127 = shadow, 255 = moving)
    cv::Mat mask;
    subtractor->apply(frame, mask);

    // 2. Mask -> binary mask: keep only strong foreground, drop the 127 shadow pixels
    cv::threshold(mask, mask, 200, 255, cv::THRESH_BINARY);

    // 3. Binary mask -> clean mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 4. Clean mask -> contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 5. Contours -> DetectedObjects
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < min_area) {
            continue;
        }
        cv::Rect box = cv::boundingRect(contour);
        // Crude confidence: how much of the frame this object covers
        float confidence = static_cast<float>(area / (frame.rows * frame.cols));
        objects.push_back({box, confidence});
    }

    return objects;
}
