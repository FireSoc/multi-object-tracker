# `detection.hpp`

POD types shared by detectors and the tracker. No translation unit; everything is in the header.

```cpp
#include "detection.hpp"
```

Depends on `<opencv2/core.hpp>` (`cv::Rect`). Also includes `<iostream>` (unused by the types themselves).

## `DetectedObject`

```cpp
struct DetectedObject {
    cv::Rect bounding_box;
    float area_confidence = 0.0;
};
```

A single axis-aligned detection in **frame pixel coordinates**, origin top-left.

This is the only type [`YoloDetector`](yolo_detector.md), [`MotionDetector`](motion_detector.md), and [`MultiObjectTracker::update`](multi_object_tracker.md) exchange. Classification is not part of the tracking contract; YOLO class scores are stuffed into `area_confidence`.

Aggregate initialization is used throughout the codebase:

```cpp
objects.push_back({box, confidence});           // MotionDetector, YoloDetector
DetectedObject d{{x, y, width, height}, 1.0F};  // tests
```

### `bounding_box`

- **Type:** `cv::Rect` (`int x, y, width, height`).
- **Units:** pixels of the `cv::Mat` that was passed to `detect`.
- **Semantics:** inclusive origin, exclusive far edge in the usual OpenCV sense (`x`..`x+width-1`).
- **Default:** default-constructed `cv::Rect` (all zeros) if omitted.

The tracker immediately converts this to `cv::Rect2f` for IoU and Kalman. Sub-pixel detector output is truncated at this field.

### `area_confidence`

- **Type:** `float`, default `0.0`.
- **Range in practice:** `[0, 1]` from both current detectors, but nothing in this struct enforces that.
- **Meaning is detector-specific** (the name is leftover from the motion path):
  - [`MotionDetector`](motion_detector.md): `contourArea / (frame.rows * frame.cols)` — fraction of the frame covered by the contour, **not** a classifier score.
  - [`YoloDetector`](yolo_detector.md): the selected class score (max over classes, or the filtered `class_id`), **after** the confidence threshold and **before** NMS copies it onto the kept box. Stored on the `DetectedObject` as `area_confidence`.
- [`MultiObjectTracker`](multi_object_tracker.md) **ignores** this field. Association is IoU-only.
- `evaluate_mot` writes this value into the MOT CSV confidence column for detections.

### Preconditions / error behavior

None at this type. Invalid boxes (`width`/`height` ≤ 0) are legal to construct. The tracker’s [`intersection_over_union`](multi_object_tracker.md#intersection_over_union) treats non-positive width or height as IoU `0`.

### Typical use

```cpp
const std::vector<DetectedObject> detections = detector.detect(frame);
const std::vector<TrackedObject>& tracks = tracker.update(detections);
```

## `ClassifiedObject`

```cpp
struct ClassifiedObject : DetectedObject {
    std::string classification;
    float classification_confidence = 0.0;
};
```

Public subclass of [`DetectedObject`](#detectedobject). **Not used** by `tracker_core`, the apps, or the tests. There is no conversion to/from `DetectedObject` vectors in the tracker; `YoloDetector` does not populate this type.

### `classification`

Label string. Unspecified vocabulary (no COCO name table in-tree). Default empty.

### `classification_confidence`

Independent of `DetectedObject::area_confidence`. Default `0.0`. No documented range.

A `ClassifiedObject` **is-a** `DetectedObject`, so it can be passed by value/reference where a `DetectedObject` is expected, but `std::vector<ClassifiedObject>` is not a `std::vector<DetectedObject>`.

## See also

- [`YoloDetector::detect`](yolo_detector.md#detect)
- [`MotionDetector::detect`](motion_detector.md#detect)
- [`TrackedObject`](multi_object_tracker.md#trackedobject) — filtered output, `cv::Rect2f`, plus identity
