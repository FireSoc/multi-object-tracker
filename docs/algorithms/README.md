# Algorithms as implemented

This tracker is a SORT-style pipeline: a detector proposes boxes, a constant-velocity Kalman filter predicts each track, Hungarian assignment matches predictions to detections by \(1 - \mathrm{IoU}\), and unmatched detections immediately become new tracks.

```
frame
  ├─ YoloDetector::detect            (ONNX, stretch-resize 640²)
  │     or MotionDetector::detect    (MOG2 foreground blobs)
  │              ↓  vector<DetectedObject>
  └─ MultiObjectTracker::update
        1. BoundingBoxKalmanFilter::predict  (all tracks)
        2. cost[i,j] = 1 - IoU(pred_i, det_j)
        3. HungarianAlgorithm::solve
        4. gate: keep assignment iff IoU ≥ 0.30
        5. matched → Kalman::update; else ++missed_frames
        6. erase tracks with missed_frames > 5
        7. unmatched dets → new Track(next_id_++)
```

Wiring: `apps/main.cpp` picks YOLO vs MOG2; `apps/evaluate_mot.cpp` always uses YOLO with COCO class 0. Association, motion, and lifecycle live entirely in `MultiObjectTracker` — the detector is a black box that emits `DetectedObject` boxes.

## File map

| Stage | Code | Doc | Reference |
|---|---|---|---|
| YOLO / ONNX / NMS | `include/yolo_detector.hpp`, `src/yolo_detector.cpp` | [detection.md](detection.md) | [yolo_detector.md](../reference/yolo_detector.md) |
| MOG2 fallback | `include/motion_detector.hpp`, `src/motion_detector.cpp` | [detection.md](detection.md) | [motion_detector.md](../reference/motion_detector.md) |
| IoU, cost, Hungarian, gating | `src/multi_object_tracker.cpp`, `src/hungarian.cpp` | [association.md](association.md) | [hungarian.md](../reference/hungarian.md), [multi_object_tracker.md](../reference/multi_object_tracker.md) |
| Kalman predict / update | `include/kalman.hpp` | [motion.md](motion.md) | [kalman.md](../reference/kalman.md) |
| Birth, coast, delete, IDs | `include/multi_object_tracker.hpp`, `src/multi_object_tracker.cpp` | [track-lifecycle.md](track-lifecycle.md) | [multi_object_tracker.md](../reference/multi_object_tracker.md) |

`DetectedObject` / `TrackedObject` are in `include/detection.hpp` and `include/multi_object_tracker.hpp`. Eigen is linked (`CMakeLists.txt`) but unused by the filter; OpenCV `cv::KalmanFilter` is the actual implementation.

## Constants that matter

| Symbol in code | Value | Role |
|---|---|---|
| `YoloDetector::kInputSize` | 640 | Network spatial size; stretch, not letterbox |
| `YoloDetector::kDefaultConfidenceThreshold` | 0.25 | Per-candidate class-score floor |
| `YoloDetector::kDefaultNmsThreshold` | 0.45 | Greedy NMS IoU |
| `kPixelScale` | \(1/255\) | Blob scale to \([0,1]\) |
| `MultiObjectTracker::kMinimumIou` | 0.30 | Post-Hungarian gate |
| `MultiObjectTracker::kMaximumMissedFrames` | 5 | Coast budget; delete when `missed_frames > 5` |
| `BoundingBoxKalmanFilter::kProcessNoise` | \(10^{-2}\) | \(Q = q I_6\) |
| `BoundingBoxKalmanFilter::kMeasurementNoise` | \(10^{-1}\) | \(R = r I_4\) |
| `BoundingBoxKalmanFilter::kInitialStateUncertainty` | \(1.0\) | \(P_{0|0} = I_6\) |
| `MotionDetector::min_area` | 500 px | Contour area floor |
| `next_id_` | starts at 1 | Monotonic, never recycled |

## What is in the code vs. not

**Present**

- Ultralytics-style YOLO ONNX via `cv::dnn::readNetFromONNX` / `forward`
- Stretch-resize preprocessing (`blobFromImage`, `crop=false`) and independent \(H/V\) unscale
- Class-agnostic or single-class score, greedy `cv::dnn::NMSBoxes`
- MOG2 + morph + contours as a no-model detector
- 6D Kalman: \([c_x, c_y, w, h, v_x, v_y]\), \(\Delta t = 1\) frame, no size velocity
- Global min-cost assignment (Kuhn–Munkres via successive shortest paths + dual potentials)
- Cost \(1 - \mathrm{IoU}\) on **predicted** boxes; IoU gate after assignment
- Immediate birth, 5-frame coast, delete on the 6th miss, IDs from 1

**Not present** (common MOT machinery this repo does not implement)

- Letterbox / affine pad-and-scale
- YOLO objectness (v8+ fused); class-aware NMS; per-class association
- SORT 7D state \((c_x, c_y, s, r, \dot{c}_x, \dot{c}_y, \dot{s})\); Mahalanobis gating
- Appearance / ReID, DeepSORT, ByteTrack high/low-score cascade
- Tentative vs confirmed tracks (`min_hits`), ID recycle, track merge
- Camera-motion compensation, occlusion flags, track-quality scores
- Pre-Hungarian forbidden edges (infinite cost); dummy “unassigned” columns
- Time-aware \(F(\Delta t)\); process-noise structure by derivative order

The public README calling this “SORT-style” is accurate at the pipeline level and false at the state-vector / confirmation-policy level. Details belong in the pages below.

## See also

- [Architecture](../architecture/README.md) — pipeline, data flow, components
- [Reference](../reference/README.md) — same types (`YoloDetector`, `HungarianAlgorithm`, `BoundingBoxKalmanFilter`, `MultiObjectTracker`)
- [Explanation index](../explanation/README.md)
- [Overview](../overview.md)
