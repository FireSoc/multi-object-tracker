# Architecture

System shape: what talks to what, which types cross boundaries, which knobs exist. Not a step list and not a Kalman derivation.

Diátaxis home for these pages: [explanation/](../explanation/README.md). Algorithm-level “why this formula” is [algorithms/](../algorithms/README.md).

## Pages

| Page | What it covers |
|---|---|
| [pipeline.md](pipeline.md) | Frame loop, `update()` stages, apps vs library, what is not in the pipeline |
| [data-flow.md](data-flow.md) | `cv::Mat` → `DetectedObject` → cost/assignment → `TrackedObject` → MOT CSV |
| [components.md](components.md) | `tracker_core` surface, ownership, runtime knobs, tests that lock contracts |

## Same types

| Topic | Here | Algorithms | Reference |
|---|---|---|---|
| Pipeline / `update` | [pipeline.md](pipeline.md) | [algorithms/](../algorithms/README.md) | [multi_object_tracker.md](../reference/multi_object_tracker.md) |
| Boundary types / capture | [data-flow.md](data-flow.md) | — | [detection.md](../reference/detection.md), [video_source.md](../reference/video_source.md) |
| Components / detection | [components.md](components.md) | [detection.md](../algorithms/detection.md) | [yolo_detector.md](../reference/yolo_detector.md), [motion_detector.md](../reference/motion_detector.md) |
| Association / Kalman / lifecycle | [pipeline.md](pipeline.md) | [association.md](../algorithms/association.md), [motion.md](../algorithms/motion.md), [track-lifecycle.md](../algorithms/track-lifecycle.md) | [hungarian.md](../reference/hungarian.md), [kalman.md](../reference/kalman.md) |

## See also

- [Algorithms](../algorithms/README.md) — SORT stages as implemented
- [Reference](../reference/README.md) — same types, signatures
- [Explanation index](../explanation/README.md)
- [Overview](../overview.md)
- [Guides](../guides/README.md) — how to run the two apps
