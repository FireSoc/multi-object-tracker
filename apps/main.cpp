#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "motion_detector.hpp"
#include "multi_object_tracker.hpp"
#include "video_source.hpp"

namespace {

constexpr int kDefaultCameraIndex = 0;
constexpr int kFrameDelayMilliseconds = 1;
constexpr int kBoxThickness = 2;
constexpr double kLabelScale = 0.6;
constexpr int kLabelThickness = 2;
constexpr char kQuitKey = 'q';
const cv::Scalar kTrackColor(0, 255, 0);

}  // namespace

int main(int argc, char* argv[]) {
    const bool using_video_file = argc > 1;
    VideoSource source =
        using_video_file ? VideoSource(std::string(argv[1])) : VideoSource(kDefaultCameraIndex);
    if (!source.is_open()) {
        std::cerr << "Could not open "
                  << (using_video_file ? std::string("video file: ") + argv[1] : "camera") << '\n';
        return 1;
    }

    MotionDetector detector;
    MultiObjectTracker tracker;

    cv::Mat frame;
    while (source.next(frame)) {
        const std::vector<DetectedObject> detections = detector.detect(frame);
        const std::vector<TrackedObject>& tracks = tracker.update(detections);

        for (const TrackedObject& track : tracks) {
            const cv::Rect box = track.bounding_box;
            cv::rectangle(frame, box, kTrackColor, kBoxThickness);

            const std::string label = "ID " + std::to_string(track.id);
            const cv::Point label_origin(box.x, std::max(0, box.y - kBoxThickness));
            cv::putText(frame, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, kLabelScale,
                        kTrackColor, kLabelThickness);
        }

        cv::imshow("tracker", frame);
        if (cv::waitKey(kFrameDelayMilliseconds) == kQuitKey) {
            break;
        }
    }
}
