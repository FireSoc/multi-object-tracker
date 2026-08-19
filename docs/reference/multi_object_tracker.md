# `multi_object_tracker.hpp`

SORT-style multi-object tracker: per-track [`BoundingBoxKalmanFilter`](kalman.md), Hungarian matching on `1 − IoU`, IoU gate, miss-based deletion, immediate spawn.

```cpp
#include "multi_object_tracker.hpp"
```

Public header also includes `detection.hpp` and `kalman.hpp`. Implementation `src/multi_object_tracker.cpp` additionally includes `hungarian.hpp`.

There is no `tracker.hpp`.

## `TrackedObject`

```cpp
struct TrackedObject {
    int id;
    cv::Rect2f bounding_box;
    int missed_frames;
};
```

Public snapshot of one live track **after** a given [`update`](#update). Aggregate, no methods.

### `id`

Stable identity. Assigned from `next_id_` starting at **`1`**, incremented on each spawn, **never reused** after the track is dropped.

### `bounding_box`

`cv::Rect2f` in frame pixels (top-left + size), from the Kalman filter:

- Spawn frame: the detection rectangle (converted `cv::Rect` → `cv::Rect2f`), **no** predict/correct yet.
- Matched frame: `filter.update(detection)`.
- Missed frame: `filter.predict()` only (coast).

Can leave the image; there is no clamp. Width/height are `≥ 0` via [`to_box`](kalman.md#to_box).

`apps/main.cpp` assigns this to `cv::Rect` for drawing (truncates floats).

### `missed_frames`

Consecutive frames since the last **accepted** match (`IoU ≥ kMinimumIou`). `0` if updated this frame. Tracks with `missed_frames > kMaximumMissedFrames` are erased **before** the snapshot is built, so you never observe a value `> 5` in the returned vector. You **do** observe `1…5` while coasting.

`evaluate_mot` still emits coasting boxes (confidence `1.0` in the CSV, not a real score).

---

## `MultiObjectTracker`

```cpp
class MultiObjectTracker {
   public:
    static constexpr double kMinimumIou = 0.30;
    static constexpr int kMaximumMissedFrames = 5;

    const std::vector<TrackedObject>& update(const std::vector<DetectedObject>& detections);

   private:
    struct Track { /* … */ };
    static double intersection_over_union(const cv::Rect2f& first, const cv::Rect2f& second);

    int next_id_ = 1;
    std::vector<Track> tracks_;
    std::vector<TrackedObject> tracked_objects_;
};
```

Default-constructible. No configuration beyond the two public constexprs (compile-time). No `reset()`; construct a new tracker.

### Public constants

| Constant | Value | Meaning |
| --- | --- | --- |
| `kMinimumIou` | `0.30` | After Hungarian, a pair is accepted only if IoU is ≥ this. Below: treated as a miss, detection remains free to spawn. |
| `kMaximumMissedFrames` | `5` | Drop when `missed_frames > 5`, i.e. the track survives **5** consecutive unmatched frames and dies on the 6th. |

These are the only gates. There is no min-hit confirmation, no max-age other than misses, no class/appearance term.

---

### `update`

```cpp
const std::vector<TrackedObject>& update(const std::vector<DetectedObject>& detections);
```

**Purpose.** One tracker timestep. Mutates internal tracks; returns a view of the survivors.

**Parameters.**

| Name | Role |
| --- | --- |
| `detections` | This frame’s boxes. [`area_confidence`](detection.md) is ignored. Empty is valid (all tracks miss). Order does not matter; IDs are preserved across permutation (see tests). |

Integer `cv::Rect` boxes are converted with `cv::Rect2f(detection.bounding_box)`.

**Returns.** `const std::vector<TrackedObject>&` bound to `tracked_objects_`.

**Lifetime.** Invalidated by the next `update` (`clear` + refill). Copy the vector or consume it before the next call. Holding the reference across frames is a bug.

**Order.** Current `tracks_` order: remaining old tracks (after `erase_if`, original relative order) then newly spawned tracks in detection-index order.

**Error behavior.** Does not throw on its own. [`HungarianAlgorithm::solve`](hungarian.md) would throw if costs were non-finite; IoU-derived costs are in `[0, 1]` so that path is clean unless you somehow pass NaN boxes.

**Preconditions.** None. First call with detections spawns a track per detection. First call with `{}` returns empty.

**Postconditions.** `tracks_.size() == tracked_objects_.size()`. Every returned `id` is unique and `≥ 1`.

---

## Frame algorithm (`src/multi_object_tracker.cpp`)

Exact sequence:

1. **Predict.** For every `Track`, `predicted_box = filter.predict()`.

2. **Assign.** If both `tracks_` and `detections` are non-empty, build

   `costs[t][d] = 1.0 - intersection_over_union(predicted_t, Rect2f(det_d))`

   and `assignments = HungarianAlgorithm::solve(costs)`.  
   Otherwise `assignments` is all `-1` (size = `#tracks`).

3. **Gate / update / miss.** For each track, `detection_index = assignments[t]`:

   - If index is in range **and** `IoU(predicted_box, det) ≥ 0.30`:  
     `predicted_box = filter.update(det)`, `missed_frames = 0`, mark detection matched.
   - Else: `++missed_frames` (no `update`; box stays at predict).

   Hungarian matches with IoU `< 0.30` are rejected; that detection is **not** marked matched and can spawn a new ID.

4. **Delete.** `std::erase_if(tracks_, missed_frames > kMaximumMissedFrames)`.

5. **Spawn.** Every unmatched detection: `tracks_.emplace_back(next_id_++, det.bounding_box)`.  
   New `Track` constructor sets `predicted_box = initial_box`, `missed_frames = 0`.

6. **Publish.** Clear `tracked_objects_`, reserve, copy `{id, predicted_box, missed_frames}` for every remaining track. Return it.

**Not SORT-complete.** No “tentative / confirmed” two-stage tracks. Every unmatched detection is a new ID immediately (false positives become tracks for up to 6 frames). No ReID / appearance. No NMS across tracks.

**Complexity.** Predict `O(T)`. Cost matrix `O(T D)` IoU tests. Hungarian `O(min(T,D) · max(T,D)²)` as in [`hungarian.md`](hungarian.md). Spawn/delete linear. `T` and `D` are small (tens).

**Memory.** `tracks_` holds a Kalman filter each. `tracked_objects_` is rebuilt every frame (no in-place mutation of previously returned elements you might have copied).

**Threading.** Not safe. One thread per tracker instance.

---

### `Track` (private nested)

```cpp
struct Track {
    Track(int track_id, const cv::Rect2f& initial_box)
        : id(track_id), filter(initial_box), predicted_box(initial_box) {}

    int id;
    BoundingBoxKalmanFilter filter;
    cv::Rect2f predicted_box;
    int missed_frames = 0;
};
```

Internal identity + filter. `predicted_box` is both the last predict/update result and the box copied into [`TrackedObject`](#trackedobject). Callers never see `Track`.

`cv::Rect` detections convert to `cv::Rect2f` via `Track`’s constructor argument (`emplace_back(next_id_++, detections[i].bounding_box)`).

---

### `intersection_over_union` (private static)

```cpp
static double intersection_over_union(const cv::Rect2f& first, const cv::Rect2f& second);
```

```
if any of first/second width or height <= 0: return 0
intersection = first & second          // cv::Rect2f intersection
union = first.area() + second.area() - intersection.area()
return union > 0 ? intersection / union : 0
```

Used for cost `1 - IoU` and for the 0.30 gate. Result in `[0, 1]`. Degenerate boxes → `0` cost `1` (worst).

OpenCV `&` on `Rect2f` handles non-overlap with zero area.

---

### Private state

| Member | Init | Role |
| --- | --- | --- |
| `next_id_` | `1` | Next spawn ID |
| `tracks_` | empty | Live filters |
| `tracked_objects_` | empty | Last published snapshot |

---

## Typical use

```cpp
MultiObjectTracker tracker;
while (source.next(frame)) {
    const auto detections = detector.detect(frame);
    const std::vector<TrackedObject>& tracks = tracker.update(detections);
    for (const TrackedObject& t : tracks) {
        cv::rectangle(frame, t.bounding_box, color, 2);
    }
}
```

Tests (`tests/multi_object_tracker_test.cpp`) cover: empty update, ID stability under detection reorder, drop after 5 misses, IoU gate spawning a second ID instead of stealing.

## Interaction with detectors

| Detector | Effect on IDs |
| --- | --- |
| [`YoloDetector`](yolo_detector.md) | Class-filtered boxes (e.g. person-only) reduce spurious spawns |
| [`MotionDetector`](motion_detector.md) | Blob merge/split and MOG2 warm-up cause extra IDs; tracker will not merge overlapping blobs |

The tracker never reads class. Filtering belongs in the detector (`YoloDetector` `class_id`).

## See also

- [`DetectedObject`](detection.md#detectedobject) — input
- [`BoundingBoxKalmanFilter`](kalman.md)
- [`HungarianAlgorithm::solve`](hungarian.md#solve)
