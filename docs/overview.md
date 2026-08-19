# Project overview

Online multi-object tracker: every frame, take axis-aligned boxes, emit the same objects with stable integer IDs. Association is SORT-class: constant-velocity Kalman on box geometry, Hungarian assignment on `1 - IoU`, then a hard IoU gate. No appearance embedding, no ByteTrack second-stage low-confidence matching, no camera-motion compensation.

Detection is pluggable and lives *outside* the tracker. `MultiObjectTracker::update` consumes `std::vector<DetectedObject>` and returns `const std::vector<TrackedObject>&`. That is the whole runtime contract.

## What it is (and is not)

**Is:** a C++20 library (`tracker_core`) plus two apps. One app (`tracker`) is an OpenCV HighGUI loop. The other (`evaluate_mot`) is a headless MOTChallenge CSV writer used by `evaluation/evaluate.py`.

**Is not:** a Python tracker, an Ultralytics `model.track()` wrapper, or a MOT17 benchmark submission. The published MOT17-05-FRCNN numbers in the repo README are a 100-frame integration baseline (COCO `person` only, YOLO11n, no MOT-specific detector training).

## Runtime data flow

```
VideoSource.next(cv::Mat)
        │
        ▼
MotionDetector.detect  ── or ──  YoloDetector.detect
        │                              │
        └──────────┬───────────────────┘
                   ▼
        std::vector<DetectedObject>
                   │
                   ▼
        MultiObjectTracker::update
                   │
                   ▼
        std::vector<TrackedObject>   // id, bounding_box, missed_frames
```

`apps/main.cpp` constructs **either** `MotionDetector` **or** `YoloDetector`, never both. `--yolo` selects YOLO; otherwise MOG2. The tracker does not know which detector ran.

`apps/evaluate_mot.cpp` always uses `YoloDetector` with `class_id = 0` (COCO person) and writes two MOT CSVs (raw detections with `id = -1`, tracks with Kalman boxes).

## Components

### Types — `include/detection.hpp`

- `DetectedObject`: `cv::Rect bounding_box`, `float area_confidence`. YOLO stores class score in `area_confidence`; MOG2 stores contour-area / frame-area.
- `ClassifiedObject`: extends `DetectedObject` with `classification` + `classification_confidence`. **Unused** by `tracker_core` and both apps.
- `TrackedObject` (in `include/multi_object_tracker.hpp`): `int id`, `cv::Rect2f bounding_box`, `int missed_frames`.

IDs start at `1` (`next_id_`).

### Capture — `VideoSource`

[`include/video_source.hpp`](../include/video_source.hpp) / [`src/video_source.cpp`](../src/video_source.cpp)

Thin `cv::VideoCapture` wrapper. `VideoSource(int camera_index = 0)` or `VideoSource(const std::string& video_path)`. `is_open()` / `next(cv::Mat&)`. Default camera index in the app is `0`.

### Detectors

**`MotionDetector`** — [`include/motion_detector.hpp`](../include/motion_detector.hpp) / [`src/motion_detector.cpp`](../src/motion_detector.cpp)

- `cv::createBackgroundSubtractorMOG2()`
- Keep mask `> 200` (drop shadow `127`)
- Ellipse 5×5 open + close
- External contours, `min_area = 500`
- Box = `cv::boundingRect`

No-model fallback for `tracker` without `--yolo`.

**`YoloDetector`** — [`include/yolo_detector.hpp`](../include/yolo_detector.hpp) / [`src/yolo_detector.cpp`](../src/yolo_detector.cpp)

- `cv::dnn::readNetFromONNX`
- Letterbox-free `blobFromImage` to **640×640**, scale `1/255`, RGB swap
- Ultralytics-style output: `[1, features, candidates]` or `[1, candidates, features]`; features = 4 box coords + class scores (no objectness)
- Defaults: `kDefaultConfidenceThreshold = 0.25`, `kDefaultNmsThreshold = 0.45`
- Optional `class_id` filter; `evaluate_mot` passes `0`
- Boxes rescaled to frame size, NMS via `cv::dnn::NMSBoxes`
- Class label is discarded; tracker never sees it

Expects a YOLOv8/YOLO11 **detection** ONNX. No weights in git (`data/*.onnx` gitignored). Export is a Python/uv concern, not CMake.

### Motion model — `BoundingBoxKalmanFilter`

[`include/kalman.hpp`](../include/kalman.hpp) — header-only.

OpenCV `cv::KalmanFilter`, **not** Eigen (Eigen is still a `tracker_core` link dependency and is only exercised by `tests/sanity_test.cpp`).

State (6): `(cx, cy, w, h, vx, vy)`. Measurement (4): `(cx, cy, w, h)`. `F` has `dt = 1` on the velocity terms. Process noise `1e-2`, measurement noise `1e-1`. Predict before assignment; `correct` only on an accepted match.

### Association — `HungarianAlgorithm` + `MultiObjectTracker`

[`include/hungarian.hpp`](../include/hungarian.hpp) / [`src/hungarian.cpp`](../src/hungarian.cpp)

Jonker–Volgenant-style min-cost assignment on a rectangular finite cost matrix. Returns one column index per row; `-1` if that row is unmatched (more rows than columns). Throws `std::invalid_argument` on ragged matrices or non-finite costs. Tall matrices are transposed internally.

[`include/multi_object_tracker.hpp`](../include/multi_object_tracker.hpp) / [`src/multi_object_tracker.cpp`](../src/multi_object_tracker.cpp)

Per `update(detections)`:

1. Kalman predict every live track.
2. Cost `1 - IoU(predicted_box, detection)`.
3. Hungarian.
4. Accept assignment only if IoU ≥ `kMinimumIou` (`0.30`); else count a miss.
5. Delete tracks with `missed_frames > kMaximumMissedFrames` (`5`) — so a track is emitted through five empty frames, gone on the sixth.
6. Unmatched detections spawn new tracks (filter initialized on the detection box).
7. Output boxes are the posterior (or predicted, if missed) `cv::Rect2f`.

Tests: [`tests/multi_object_tracker_test.cpp`](../tests/multi_object_tracker_test.cpp), [`tests/hungarian_test.cpp`](../tests/hungarian_test.cpp).

## Languages and tooling

| Layer | Language | Role |
|---|---|---|
| `include/`, `src/`, `apps/` | C++20 (`CMAKE_CXX_STANDARD 20`, extensions off) | Tracker and CLIs |
| `examples/*.cpp` | C++ | OpenCV practice binaries; GLOB’d into extra executables; **not** `tracker_core` |
| `evaluation/evaluate.py` | Python ≥ 3.12 | Download MOT17-05 subset, build `evaluate_mot`, score with `motmetrics` |
| `tests/test_evaluation.py` | Python | Unit tests for IoU matching / ignore-class suppression / MOT parse |
| `pyproject.toml` + `uv.lock` | uv | Ultralytics, ONNX, `motmetrics`, `numpy<2`, pytest — **not** OpenCV/Eigen |

C++ deps (`CMakeLists.txt`): OpenCV (`core`, `imgproc`, `imgcodecs`, `videoio`, `highgui`, `dnn`, `video`, `geometry`), Eigen3 (`find_package(Eigen3 REQUIRED NO_MODULE)`). GoogleTest v1.15.2 via FetchContent when `BUILD_TESTS` is ON (default).

`CMAKE_EXPORT_COMPILE_COMMANDS` is ON. Default `CMAKE_BUILD_TYPE` is Debug if unset. `-Wall -Wextra -Wpedantic`.

## Build artifacts

From [`CMakeLists.txt`](../CMakeLists.txt) / [`CMakePresets.json`](../CMakePresets.json):

| Target | Source | Purpose |
|---|---|---|
| `tracker_core` | `src/video_source.cpp`, `motion_detector.cpp`, `yolo_detector.cpp`, `hungarian.cpp`, `multi_object_tracker.cpp` | Public headers under `include/` |
| `tracker` | `apps/main.cpp` | HighGUI loop; quit key `q` |
| `evaluate_mot` | `apps/evaluate_mot.cpp` | `evaluate_mot <model.onnx> <image-dir> <detections.csv> <tracks.csv> <frame-count>` |
| `<stem>` per `examples/*.cpp` | e.g. `shapes`, `opencv_prac`, `img_transform` | Linked to OpenCV+Eigen only |
| `tracker_tests` | `tests/sanity_test.cpp`, `hungarian_test.cpp`, `multi_object_tracker_test.cpp` | `gtest_discover_tests` |

Binary locations:

- `cmake -S . -B build` → `build/tracker`, `build/evaluate_mot`, `build/tracker_tests`
- `--preset debug` / `release` → `build/debug/` or `build/release/` (evaluation uses debug)

Python does not install a package. `uv run python evaluation/evaluate.py` expects `data/yolo11n.onnx` and writes under gitignored `data/mot17/` (`MOT17-05-FRCNN` frames + `gt/gt.txt` from the Hugging Face mirror pinned in `evaluate.py`, plus `data/mot17/results/{detections,tracks}.txt`).

## Interactive CLI (`tracker`)

From `apps/main.cpp`:

```
tracker
tracker <video-path>
tracker --yolo <model.onnx>
tracker --yolo <model.onnx> <video-path>
```

Draws green `cv::rectangle` + `"ID " + id`. Frame delay 1 ms.

## Evaluation loop (not the tracker)

`evaluation/evaluate.py`: frames 1–100 of MOT17-05-FRCNN, IoU 0.50, pedestrian GT class 1, ignore classes `{2, 7, 8, 12}` after protecting pedestrian matches, no visibility cutoff. Detection precision/recall is Hungarian-at-0.50 on YOLO boxes; MOTA/IDF1/FP/FN/IDSW from `motmetrics` on tracker boxes (including the five-frame coast).

That protocol’s *rationale* is in [architecture/data-flow.md](architecture/data-flow.md); flags and column layout are indexed in [reference/](reference/README.md); the commands are in [guides/evaluation.md](guides/evaluation.md).

## What’s next

- Navigation: [docs/README.md](README.md)
- Authoring rules: [CONVENTIONS.md](CONVENTIONS.md)
- First run: [tutorials/first-run.md](tutorials/first-run.md)
- Build / export / eval commands: [guides/](guides/README.md)
- Types and CLIs: [reference/](reference/README.md)
- Why the pipeline is this shape: [explanation/](explanation/README.md)
