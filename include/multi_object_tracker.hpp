#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "detection.hpp"
#include "kalman.hpp"

struct TrackedObject {
    int id;
    cv::Rect2f bounding_box;
    int missed_frames;
};

class MultiObjectTracker {
   public:
    static constexpr double kMinimumIou = 0.30;
    static constexpr int kMaximumMissedFrames = 5;

    const std::vector<TrackedObject>& update(const std::vector<DetectedObject>& detections);

   private:
    struct Track {
        Track(int track_id, const cv::Rect2f& initial_box)
            : id(track_id), filter(initial_box), predicted_box(initial_box) {}

        int id;
        BoundingBoxKalmanFilter filter;
        cv::Rect2f predicted_box;
        int missed_frames = 0;
    };

    static double intersection_over_union(const cv::Rect2f& first, const cv::Rect2f& second);

    int next_id_ = 1;
    std::vector<Track> tracks_;
    std::vector<TrackedObject> tracked_objects_;
};
