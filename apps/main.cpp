// Entry point for the multi-object tracker.
// For now this is just a smoke test that OpenCV and Eigen link correctly.

#include <iostream>
#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "video_source.hpp"


int main() {
    VideoSource source(0);
    if (!source.is_open()) {
        std::cerr << "Could not open webcam\n";
        return 1;
    }

    cv::Mat frame;
    while (source.next(frame)) {
        cv::imshow("tracker", frame);
        if (cv::waitKey(1) == 'q') break;
    }
}
