# Reference

Information-oriented. Types, invariants, CLIs, file formats, build targets. No numbered “now run this” except a see-also to [guides](../guides/README.md).

Style: synopsis → paths → parameters / errors → see-also. Match identifiers in headers exactly. One page per public header; filename is the header stem (`hungarian.md` ← `include/hungarian.hpp`).

## C++ modules

| Page | Header | What it is |
|---|---|---|
| [detection.md](detection.md) | `include/detection.hpp` | `DetectedObject`, unused `ClassifiedObject` |
| [video_source.md](video_source.md) | `include/video_source.hpp` | `VideoSource` over `cv::VideoCapture` |
| [motion_detector.md](motion_detector.md) | `include/motion_detector.hpp` | MOG2, `min_area = 500`, mask threshold 200 |
| [yolo_detector.md](yolo_detector.md) | `include/yolo_detector.hpp` | 640, conf 0.25, NMS 0.45, ONNX layout, `class_id` |
| [kalman.md](kalman.md) | `include/kalman.hpp` | `BoundingBoxKalmanFilter`: OpenCV KF, 6-state / 4-measure |
| [hungarian.md](hungarian.md) | `include/hungarian.hpp` | `HungarianAlgorithm::solve`: rectangular, finite, `-1` unmatched |
| [multi_object_tracker.md](multi_object_tracker.md) | `include/multi_object_tracker.hpp` | `MultiObjectTracker`, `TrackedObject`, `kMinimumIou = 0.30`, `kMaximumMissedFrames = 5` |

## CLI

Exact usage strings from `apps/`. Recipes: [guides/run.md](../guides/run.md).

### `tracker`

```
Usage: tracker [<video-path> | --yolo <model.onnx> [<video-path>]]
```

`apps/main.cpp`. Camera index hardcoded to `0`. `--yolo` must be argv[1]. Window name `tracker`; quit key `q`. No `--help`, no conf/NMS/class flags. Live YOLO is all classes; `evaluate_mot` is person-only.

### `evaluate_mot`

```
Usage: evaluate_mot <model.onnx> <image-dir> <detections.csv> <tracks.csv> <frame-count>
```

`apps/evaluate_mot.cpp`. Five operands, `argc == 6`. YOLO: 640, conf 0.25, NMS 0.45, COCO class 0. Writes two MOT CSVs. Does not score.

Python wrapper flags (`--frames`, `--iou`) and the download/build/score loop: [guides/evaluation.md](../guides/evaluation.md). `evaluation/evaluate.py` constants (`SEQUENCE`, `FRAME_COUNT`) and helpers: [guides/python.md](../guides/python.md).

## MOT CSV

MOT Challenge, comma-separated, no header. Full parse/score rules: [guides/evaluation.md](../guides/evaluation.md).

```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>[, <class>, <visibility>[, ...]]
```

`evaluate_mot` `write_box`: 1-based `x,y` (`OpenCV + 1`), 3 decimal places, ten columns. Detection `id` is `-1`; track `id` is the SORT id; track conf is `1.0`; trailing columns `-1`. GT parse (`parse_mot(..., ground_truth=True)`) reads class/visibility from columns 8–9.

## CMake targets

From `CMakeLists.txt` / `CMakePresets.json`. How to configure: [guides/build.md](../guides/build.md).

| Target | Source | Purpose |
|---|---|---|
| `tracker_core` | `src/*.cpp` + `include/` | Public library. `kalman.hpp` is header-only. |
| `tracker` | `apps/main.cpp` | HighGUI loop |
| `evaluate_mot` | `apps/evaluate_mot.cpp` | Headless MOT writer |
| `<stem>` per `examples/*.cpp` | e.g. `shapes` | OpenCV practice; not `tracker_core` |
| `tracker_tests` | `tests/*.cpp` | GoogleTest when `BUILD_TESTS=ON` (default) |

OpenCV components: `core`, `imgproc`, `imgcodecs`, `videoio`, `highgui`, `dnn`, `video`, `geometry`. Eigen3 is linked; the Kalman filter does **not** use it (`tests/sanity_test.cpp` only).

Output dirs: `cmake -S . -B build` → `build/`. `--preset debug` / `release` → `build/debug/` / `build/release/`. Evaluation uses the debug preset.

## Same types (architecture / algorithms)

| Topic | Architecture | Algorithms | This folder |
|---|---|---|---|
| Pipeline / `update` | [pipeline.md](../architecture/pipeline.md) | [algorithms/](../algorithms/README.md) | [multi_object_tracker.md](multi_object_tracker.md) |
| Boundary types | [data-flow.md](../architecture/data-flow.md) | — | [detection.md](detection.md) |
| Capture | [data-flow.md](../architecture/data-flow.md) | — | [video_source.md](video_source.md) |
| Detection | [components.md](../architecture/components.md) | [detection.md](../algorithms/detection.md) | [yolo_detector.md](yolo_detector.md), [motion_detector.md](motion_detector.md) |
| Association | [pipeline.md](../architecture/pipeline.md) | [association.md](../algorithms/association.md) | [hungarian.md](hungarian.md) |
| Kalman | [pipeline.md](../architecture/pipeline.md) | [motion.md](../algorithms/motion.md) | [kalman.md](kalman.md) |
| Lifecycle | [pipeline.md](../architecture/pipeline.md) | [track-lifecycle.md](../algorithms/track-lifecycle.md) | [multi_object_tracker.md](multi_object_tracker.md) |

## Do not put here

Why `dt = 1` on velocity, why IoU gate is 0.30, why evaluation ignores MOT classes 2/7/8/12 — [explanation](../explanation/README.md).

## See also

- [How-to / guides](../guides/README.md)
- [Explanation](../explanation/README.md)
- [Overview](../overview.md)
