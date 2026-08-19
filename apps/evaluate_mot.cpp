#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "multi_object_tracker.hpp"
#include "yolo_detector.hpp"

namespace {

constexpr int kCocoPersonClassId = 0;

void write_box(std::ofstream& output, int frame, int id, const cv::Rect2f& box, float confidence) {
    output << frame << ',' << id << ',' << box.x + 1.0F << ',' << box.y + 1.0F << ',' << box.width
           << ',' << box.height << ',' << confidence << ",-1,-1,-1\n";
}

std::filesystem::path frame_path(const std::filesystem::path& image_directory, int frame) {
    std::ostringstream filename;
    filename << std::setfill('0') << std::setw(6) << frame << ".jpg";
    return image_directory / filename.str();
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: evaluate_mot <model.onnx> <image-dir> <detections.csv> "
                     "<tracks.csv> <frame-count>\n";
        return 1;
    }

    try {
        const std::filesystem::path model_path(argv[1]);
        const std::filesystem::path image_directory(argv[2]);
        const std::filesystem::path detection_path(argv[3]);
        const std::filesystem::path track_path(argv[4]);
        const int frame_count = std::stoi(argv[5]);
        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::runtime_error("YOLO model not found: " + model_path.string());
        }
        if (!std::filesystem::is_directory(image_directory)) {
            throw std::runtime_error("MOT image directory not found: " + image_directory.string());
        }
        if (frame_count <= 0) {
            throw std::invalid_argument("Frame count must be positive");
        }

        std::filesystem::create_directories(detection_path.parent_path());
        std::filesystem::create_directories(track_path.parent_path());
        std::ofstream detections_output(detection_path);
        std::ofstream tracks_output(track_path);
        if (!detections_output || !tracks_output) {
            throw std::runtime_error("Could not open evaluation output files");
        }
        detections_output << std::fixed << std::setprecision(3);
        tracks_output << std::fixed << std::setprecision(3);

        YoloDetector detector(model_path.string(), YoloDetector::kDefaultConfidenceThreshold,
                              YoloDetector::kDefaultNmsThreshold, kCocoPersonClassId);
        MultiObjectTracker tracker;

        for (int frame_number = 1; frame_number <= frame_count; ++frame_number) {
            const std::filesystem::path image_path = frame_path(image_directory, frame_number);
            const cv::Mat frame = cv::imread(image_path.string());
            if (frame.empty()) {
                throw std::runtime_error("Could not read MOT frame: " + image_path.string());
            }

            const std::vector<DetectedObject> detections = detector.detect(frame);
            for (const DetectedObject& detection : detections) {
                write_box(detections_output, frame_number, -1, detection.bounding_box,
                          detection.area_confidence);
            }

            const std::vector<TrackedObject>& tracks = tracker.update(detections);
            for (const TrackedObject& track : tracks) {
                write_box(tracks_output, frame_number, track.id, track.bounding_box, 1.0F);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Evaluation error: " << error.what() << '\n';
        return 1;
    }
}
