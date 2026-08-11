// Entry point for the multi-object tracker.
// For now this is just a smoke test that OpenCV and Eigen link correctly.

#include <iostream>
#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "video_source.hpp"
#include "motion_detector.hpp"


int main() {
    
    VideoSource source(0);
    if (!source.is_open()) {
        std::cerr << "Could not open webcam\n";
        return 1;
    }

    MotionDetector detector;

    cv::Mat frame;
    while (source.next(frame)) {

        auto objects = detector.detect(frame);
        for (const auto& obj : objects) {
            cv::rectangle(frame, obj.bounding_box, {0, 255, 0}, 2);
        }

        cv::imshow("tracker", frame);
        if (cv::waitKey(1) == 'q') break;
    }
}
