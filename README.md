# Multi-Object Tracker

OpenCV-based object detection with SORT-style multi-object tracking.

## Documentation

[docs/README.md](docs/README.md) — tutorials, how-to guides, API/CLI reference, and design explanation.

## Debug build and tests

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

## YOLO model export

Install the locked Python tooling into `.venv`:

```sh
uv sync
```

Export an Ultralytics model to ONNX:

```sh
uv run yolo export model=data/yolo11n.pt format=onnx imgsz=640 opset=12 simplify=True
```

uv manages only the Python model-export dependencies. CMake and system package
management remain responsible for the C++ OpenCV and Eigen dependencies.

## Usage

```sh
./build/tracker
./build/tracker <video-path>
./build/tracker --yolo <model.onnx>
./build/tracker --yolo <model.onnx> <video-path>
```

- `tracker`: webcam + MOG2
- `tracker <video-path>`: video + MOG2
- `tracker --yolo <model.onnx>`: webcam + YOLO
- `tracker --yolo <model.onnx> <video-path>`: video + YOLO

YOLO expects an Ultralytics YOLOv8 or YOLO11 detection model exported to ONNX.
No model is bundled.
Detection runs on the original frames, then the existing Kalman + IoU +
Hungarian SORT pipeline assigns track IDs. MOG2 remains the no-model fallback.

## Evaluation

The reproducible smoke evaluation uses frames 1–100 of the public MOT17 training
sequence `MOT17-05-FRCNN` (640x480). The script selectively downloads 3.82 MiB
of images and the official `gt/gt.txt` from the
[`Lekim89/MOT17`](https://huggingface.co/datasets/Lekim89/MOT17) mirror, pinned
to commit `f93908467986c77765667925a526ad07be8ad630`. Data and generated MOT CSV
files stay under the gitignored `data/mot17/`.

From the repository root:

```sh
uv run python evaluation/evaluate.py
```

This validates `data/yolo11n.onnx`, builds the debug `evaluate_mot` executable,
and runs OpenCV DNN YOLO11n at 640x640 with confidence 0.25 and NMS 0.45. Only
COCO class 0 (`person`) is sent to the existing C++ `MultiObjectTracker`.
Detection matching and MOT association both use one-to-one matching at IoU
0.50.

Measured on frames 1–100:

- Precision: 0.7217
- Recall: 0.4706
- MOTA: 0.1151
- IDF1: 0.4680
- Tracking false positives: 330
- Tracking false negatives: 364
- ID switches: 13

Ground truth includes every marked (`conf=1`) class-1 pedestrian regardless of
visibility; there is no visibility cutoff. Predictions associated with MOT17
person-on-vehicle, static-person, distractor, or reflection regions (classes 2,
7, 8, and 12) at IoU 0.50 are ignored after protecting pedestrian matches.
Other non-pedestrian GT classes are excluded. Tracker boxes predicted through
misses are emitted and counted for up to the tracker's five-frame miss window.

This 100-frame, single-sequence result is an integration baseline, not a
benchmark claim or evidence of generalization across MOT17. It uses a small
generic COCO model, no MOT-specific detector tuning, and the project's minimal
SORT implementation.
