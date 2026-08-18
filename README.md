# Multi-Object Tracker

OpenCV-based object detection with SORT-style multi-object tracking.

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
