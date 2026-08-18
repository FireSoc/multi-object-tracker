#include "multi_object_tracker.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "hungarian.hpp"

const std::vector<TrackedObject>& MultiObjectTracker::update(
    const std::vector<DetectedObject>& detections) {
    for (Track& track : tracks_) {
        track.predicted_box = track.filter.predict();
    }

    std::vector<int> assignments(tracks_.size(), -1);
    if (!tracks_.empty() && !detections.empty()) {
        HungarianAlgorithm::CostMatrix costs(tracks_.size(),
                                             std::vector<double>(detections.size()));

        for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
            for (std::size_t detection_index = 0; detection_index < detections.size();
                 ++detection_index) {
                costs[track_index][detection_index] =
                    1.0 -
                    intersection_over_union(tracks_[track_index].predicted_box,
                                            cv::Rect2f(detections[detection_index].bounding_box));
            }
        }
        assignments = HungarianAlgorithm::solve(costs);
    }

    std::vector<bool> matched_detections(detections.size(), false);
    for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
        Track& track = tracks_[track_index];
        const int detection_index = assignments[track_index];
        const bool has_assignment =
            detection_index >= 0 && static_cast<std::size_t>(detection_index) < detections.size();

        if (has_assignment) {
            const cv::Rect2f detection_box =
                detections[static_cast<std::size_t>(detection_index)].bounding_box;
            const double iou = intersection_over_union(track.predicted_box, detection_box);
            if (iou >= kMinimumIou) {
                track.predicted_box = track.filter.update(detection_box);
                track.missed_frames = 0;
                matched_detections[static_cast<std::size_t>(detection_index)] = true;
                continue;
            }
        }

        ++track.missed_frames;
    }

    std::erase_if(tracks_,
                  [](const Track& track) { return track.missed_frames > kMaximumMissedFrames; });

    for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
        if (!matched_detections[detection_index]) {
            tracks_.emplace_back(next_id_++, detections[detection_index].bounding_box);
        }
    }

    tracked_objects_.clear();
    tracked_objects_.reserve(tracks_.size());
    for (const Track& track : tracks_) {
        tracked_objects_.push_back({track.id, track.predicted_box, track.missed_frames});
    }
    return tracked_objects_;
}

double MultiObjectTracker::intersection_over_union(const cv::Rect2f& first,
                                                   const cv::Rect2f& second) {
    if (first.width <= 0.0F || first.height <= 0.0F || second.width <= 0.0F ||
        second.height <= 0.0F) {
        return 0.0;
    }

    const cv::Rect2f intersection = first & second;
    const double intersection_area = intersection.area();
    const double union_area = first.area() + second.area() - intersection_area;
    return union_area > 0.0 ? intersection_area / union_area : 0.0;
}
