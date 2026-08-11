// Struct for detecting an Object
#pragma once

#include <iostream>
#include <opencv2/core.hpp>

struct DetectedObject {
    // contains width, height, coords, .contains, etc.
    cv::Rect bounding_box;
    float area_confidence = 0.0;
};

struct ClassifiedObject : DetectedObject {

    std::string classification;
    float classification_confidence = 0.0;

};