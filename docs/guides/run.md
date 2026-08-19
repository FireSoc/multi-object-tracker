# Run

Binaries: `tracker` (GUI loop) and `evaluate_mot` (headless MOT writer). Paths below assume the debug preset (`build/debug/`). Swap `build/` if you configured `-B build`.

## `tracker`

```
Usage: tracker [<video-path> | --yolo <model.onnx> [<video-path>]]
```

Exactly those shapes. Anything else (including a lone `--flag`) prints usage and exits 1. Camera index is hardcoded to `0`. No `--help`, no confidence/NMS/class flags.

| Invocation | Source | Detector |
| --- | --- | --- |
| `./build/debug/tracker` | camera 0 | MOG2 (`MotionDetector`) |
| `./build/debug/tracker <video-path>` | file | MOG2 |
| `./build/debug/tracker --yolo <model.onnx>` | camera 0 | YOLO, all classes |
| `./build/debug/tracker --yolo <model.onnx> <video-path>` | file | YOLO, all classes |

`--yolo` must be argv[1]. Model path and video path must be non-empty and must not start with `--`.

Pipeline: detect on the original frame → `MultiObjectTracker::update` (Kalman predict, Hungarian on `1 - IoU`, gate at IoU 0.30, miss window 5). YOLO here does **not** filter to person; `evaluate_mot` does. MOG2 is the no-model fallback: `createBackgroundSubtractorMOG2`, drop shadows (`threshold` 200), open/close ellipse 5×5, contours with `min_area` 500.

Display: window `"tracker"`, green boxes, label `ID N`. `q` quits (`waitKey(1)`). OpenCV/std exceptions go to stderr, exit 1. Failed open: `"Could not open video file: …"` / `"Could not open camera"`.

## `evaluate_mot`

```
Usage: evaluate_mot <model.onnx> <image-dir> <detections.csv> <tracks.csv> <frame-count>
```

`argc` must be 6. Positional only.

```sh
./build/debug/evaluate_mot \
  data/yolo11n.onnx \
  data/mot17/MOT17-05-FRCNN/img1 \
  data/mot17/results/detections.txt \
  data/mot17/results/tracks.txt \
  100
```

- Frames are `img1/000001.jpg` … `img1/{frame-count:06d}.jpg` (1-based, six-digit). Missing/unreadable frame is fatal.
- YOLO: OpenCV DNN ONNX, 640×640, conf 0.25, NMS 0.45, **COCO class 0 (`person`) only**.
- Creates parent dirs for the two output paths.
- Writes MOT Challenge CSV, 3 decimal places. `x`/`y` are **+1** vs OpenCV (MOT is 1-indexed). Detection `id` is `-1`; track `id` is the SORT id; track confidence is `1.0`. Trailing columns `-1,-1,-1`.
- Emits predicted boxes through the five-frame miss window (unmatched tracks stay in the dump until dropped).

Does not score. Scoring is `evaluation/evaluate.py` — [evaluation.md](evaluation.md).

## Models

Nothing is bundled. Interactive YOLO and `evaluate_mot` both call `cv::dnn::readNetFromONNX`. Expected: Ultralytics YOLOv8 / YOLO11 detection ONNX.

Export (README; weights at `data/yolo11n.pt`, gitignored):

```sh
uv sync
uv run yolo export model=data/yolo11n.pt format=onnx imgsz=640 opset=12 simplify=True
```

`evaluate.py` requires `data/yolo11n.onnx` (also gitignored). Input blob is 640×640, RGB, `/255`. Output accepted as `[1, features, candidates]` or `[1, candidates, features]` with ≥5 features (`cx, cy, w, h` + class scores).

## Examples

CMake globs `examples/*.cpp` into same-named binaries (`shapes`, `img_transform`, `opencv_prac`). Practice OpenCV, not the tracker. `img_transform` / `opencv_prac` hardcode `data/me.jpeg`.
