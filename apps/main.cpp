#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "motion_detector.hpp"
#include "multi_object_tracker.hpp"
#include "video_source.hpp"
#include "yolo_detector.hpp"

namespace {

constexpr int kDefaultCameraIndex = 0;
constexpr int kFrameDelayMilliseconds = 1;
constexpr int kBoxThickness = 2;
constexpr double kLabelScale = 0.6;
constexpr int kLabelThickness = 2;
constexpr char kQuitKey = 'q';
constexpr char kYoloOption[] = "--yolo";
constexpr char kUsage[] =
    "Usage: tracker [<video-path> | --yolo <model.onnx> [<video-path>]]";
const cv::Scalar kTrackColor(0, 255, 0);

}  // namespace

int main(int argc, char* argv[]) {
    bool use_yolo = false;
    std::string model_path;
    std::string video_path;

    if (argc == 2 &&
        (std::string(argv[1]).empty() || std::string(argv[1]).starts_with("--"))) {
        std::cerr << kUsage << '\n';
        return 1;
    }
    if (argc == 2) {
        video_path = argv[1];
    } else if ((argc == 3 || argc == 4) && std::string(argv[1]) == kYoloOption &&
               !std::string(argv[2]).empty() && !std::string(argv[2]).starts_with("--") &&
               (argc == 3 ||
                (!std::string(argv[3]).empty() && !std::string(argv[3]).starts_with("--")))) {
        use_yolo = true;
        model_path = argv[2];
        if (argc == 4) {
            video_path = argv[3];
        }
    } else if (argc != 1) {
        std::cerr << kUsage << '\n';
        return 1;
    }

    try {
        std::unique_ptr<MotionDetector> motion_detector;
        std::unique_ptr<YoloDetector> yolo_detector;
        if (use_yolo) {
            yolo_detector = std::make_unique<YoloDetector>(model_path);
        } else {
            motion_detector = std::make_unique<MotionDetector>();
        }

        const bool using_video_file = !video_path.empty();
        VideoSource source =
            using_video_file ? VideoSource(video_path) : VideoSource(kDefaultCameraIndex);
        if (!source.is_open()) {
            std::cerr << "Could not open "
                      << (using_video_file ? std::string("video file: ") + video_path : "camera")
                      << '\n';
            return 1;
        }

        MultiObjectTracker tracker;
        cv::Mat frame;
        while (source.next(frame)) {
            const std::vector<DetectedObject> detections =
                use_yolo ? yolo_detector->detect(frame) : motion_detector->detect(frame);
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
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
