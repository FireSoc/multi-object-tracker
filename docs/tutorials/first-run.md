# First run: webcam + MOG2

Goal: a Debug `tracker` binary, camera 0 (or a file), green `ID N` overlays, quit with `q`.

This is the default `apps/main.cpp` path: `MotionDetector` (MOG2), then `MultiObjectTracker::update`. No ONNX. Build flags and the other three CLI shapes live in [guides/build.md](../guides/build.md) and [guides/run.md](../guides/run.md) — do not copy those pages here.

## Prerequisites

- CMake 3.24+, C++20 toolchain, OpenCV (including `video` / `dnn` / `highgui`) and Eigen3
- A camera at index 0, **or** a video file
- Repo root as cwd

## Steps

1. Configure and build the README layout:

   ```sh
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
   cmake --build build --target tracker
   ```

   Preset layout (`build/debug/`) is in [build.md](../guides/build.md). Do not mix the two output dirs.

2. Run MOG2 on the webcam:

   ```sh
   ./build/tracker
   ```

   File instead of camera: `./build/tracker <video-path>`.

3. Confirm a HighGUI window titled `tracker`. Moving blobs should get green boxes labeled `ID N` (`N` starting at 1). MOG2 needs a few frames to learn background; a static scene may show nothing at first.

4. Press `q` to quit (`waitKey(1)` in `apps/main.cpp`). Failed open prints `Could not open camera` / `Could not open video file: …` and exits 1.

## YOLO (optional, same binary)

Needs an Ultralytics detection ONNX (not bundled). Export is [guides/python.md](../guides/python.md). Then:

```sh
./build/tracker --yolo data/yolo11n.onnx
./build/tracker --yolo data/yolo11n.onnx <video-path>
```

Live YOLO does **not** filter to COCO person; `evaluate_mot` does. Flags and ONNX layout: [run.md](../guides/run.md), [`YoloDetector`](../reference/yolo_detector.md).

## What’s next

- Other invocations and `evaluate_mot`: [guides/run.md](../guides/run.md)
- MOT17-05 smoke: [guides/evaluation.md](../guides/evaluation.md)
- Types: [reference/](../reference/README.md)
- Why this pipeline: [explanation/](../explanation/README.md)
