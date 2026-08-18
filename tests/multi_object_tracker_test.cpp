#include "multi_object_tracker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace {

DetectedObject detection(int x, int y, int width, int height) {
    return {{x, y, width, height}, 1.0F};
}

const TrackedObject& find_track(const std::vector<TrackedObject>& tracks, int id) {
    return *std::find_if(tracks.begin(), tracks.end(),
                         [id](const TrackedObject& track) { return track.id == id; });
}

}  // namespace

TEST(MultiObjectTrackerTest, HandlesFramesWithoutTracksOrDetections) {
    MultiObjectTracker tracker;

    EXPECT_TRUE(tracker.update({}).empty());
}

TEST(MultiObjectTrackerTest, KeepsIdsWhenDetectionOrderChanges) {
    MultiObjectTracker tracker;
    const auto& initial_tracks =
        tracker.update({detection(0, 0, 20, 20), detection(100, 0, 20, 20)});
    ASSERT_EQ(initial_tracks.size(), 2U);
    const int left_id = initial_tracks[0].id;
    const int right_id = initial_tracks[1].id;

    const auto& reordered_tracks =
        tracker.update({detection(100, 0, 20, 20), detection(0, 0, 20, 20)});

    ASSERT_EQ(reordered_tracks.size(), 2U);
    EXPECT_LT(find_track(reordered_tracks, left_id).bounding_box.x, 50.0F);
    EXPECT_GT(find_track(reordered_tracks, right_id).bounding_box.x, 50.0F);
}

TEST(MultiObjectTrackerTest, DropsTrackAfterMaximumMissedFrames) {
    MultiObjectTracker tracker;
    ASSERT_EQ(tracker.update({detection(0, 0, 20, 20)}).size(), 1U);

    for (int missed_frame = 0; missed_frame < MultiObjectTracker::kMaximumMissedFrames;
         ++missed_frame) {
        EXPECT_EQ(tracker.update({}).size(), 1U);
    }

    EXPECT_TRUE(tracker.update({}).empty());
}

TEST(MultiObjectTrackerTest, RejectsAssignmentsBelowMinimumIou) {
    MultiObjectTracker tracker;
    const int original_id = tracker.update({detection(0, 0, 20, 20)})[0].id;

    const auto& tracks = tracker.update({detection(100, 100, 20, 20)});

    ASSERT_EQ(tracks.size(), 2U);
    EXPECT_EQ(find_track(tracks, original_id).missed_frames, 1);
}
