# Explanation

Understanding-oriented. Why the pipeline is this shape. Cite functions; leave signatures in [reference](../reference/README.md).

Content lives in [architecture/](../architecture/README.md) (system shape) and [algorithms/](../algorithms/README.md) (SORT stages as coded). This page is the Diátaxis index, not a second copy.

Not a MOT survey. Compare to SORT/ByteTrack only where this repo’s code diverges.

## Architecture

| Page | Question it answers |
|---|---|
| [pipeline.md](../architecture/pipeline.md) | End-to-end path: apps construct concrete detectors, call `MultiObjectTracker::update`, draw or write MOT CSV. No detector ABC. |
| [data-flow.md](../architecture/data-flow.md) | Types at stage boundaries, MOT CSV / scoring data, ownership. |
| [components.md](../architecture/components.md) | What is in `tracker_core` vs apps, knobs that actually change behavior. |

## Algorithms

| Page | Question it answers | Cite |
|---|---|---|
| [README.md](../algorithms/README.md) | SORT-class vs this tree (no ByteTrack second stage, no ReID/CMC, 6-D not 7-D) | `update()`, papers only as contrast |
| [detection.md](../algorithms/detection.md) | MOG2 as no-weight fallback vs YOLO DNN; class scores never enter association | `MotionDetector`, `YoloDetector` |
| [association.md](../algorithms/association.md) | Predict → `1-IoU` Hungarian → 0.30 gate → miss counter → spawn | `src/multi_object_tracker.cpp` |
| [motion.md](../algorithms/motion.md) | Why 6-D `(cx,cy,w,h,vx,vy)` with OpenCV KF; Eigen linked but unused in the filter | `include/kalman.hpp`, `CMakeLists.txt` |
| [track-lifecycle.md](../algorithms/track-lifecycle.md) | Immediate birth, 5-frame coast, delete on miss `> 5`, IDs from 1 | `MultiObjectTracker` |

Evaluation protocol (MOT17-05-FRCNN frames 1–100, pedestrian-only, ignore `{2, 7, 8, 12}`, no visibility cutoff, coasted boxes scored) is in [data-flow.md](../architecture/data-flow.md) and the command/metric page [guides/evaluation.md](../guides/evaluation.md). There is no separate `evaluation-protocol.md`.

## Same types (architecture / algorithms / reference)

| Topic | Architecture | Algorithms | Reference |
|---|---|---|---|
| Pipeline / `update` | [pipeline.md](../architecture/pipeline.md) | [algorithms/](../algorithms/README.md) | [multi_object_tracker.md](../reference/multi_object_tracker.md) |
| Boundary types | [data-flow.md](../architecture/data-flow.md) | — | [detection.md](../reference/detection.md) |
| Capture | [data-flow.md](../architecture/data-flow.md) | — | [video_source.md](../reference/video_source.md) |
| Detection | [components.md](../architecture/components.md) | [detection.md](../algorithms/detection.md) | [yolo_detector.md](../reference/yolo_detector.md), [motion_detector.md](../reference/motion_detector.md) |
| Association | [pipeline.md](../architecture/pipeline.md) | [association.md](../algorithms/association.md) | [hungarian.md](../reference/hungarian.md) |
| Kalman | [pipeline.md](../architecture/pipeline.md) | [motion.md](../algorithms/motion.md) | [kalman.md](../reference/kalman.md) |
| Lifecycle | [pipeline.md](../architecture/pipeline.md) | [track-lifecycle.md](../algorithms/track-lifecycle.md) | [multi_object_tracker.md](../reference/multi_object_tracker.md) |

## See also

- [Architecture index](../architecture/README.md)
- [Algorithms index](../algorithms/README.md)
- [Reference](../reference/README.md) — constants and APIs
- [Overview](../overview.md) — system map
