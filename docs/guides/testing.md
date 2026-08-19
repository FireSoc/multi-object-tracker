# Testing

Two harnesses: GoogleTest for C++ (`tracker_tests`), pytest for MOT parsing/matching. Neither substitutes for the other. Default CMake `BUILD_TESTS=ON`.

## C++ (GoogleTest 1.15.2)

Configure + build, then either:

```sh
# presets (matches .clangd / evaluate.py)
cmake --preset debug
cmake --build --preset debug --target tracker_tests
ctest --preset debug
```

```sh
# README layout
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target tracker_tests
ctest --test-dir build --output-on-failure
```

`gtest_discover_tests(tracker_tests DISCOVERY_MODE PRE_TEST)` — ctest enumerates gtest cases at test time. Direct:

```sh
./build/debug/tracker_tests
# or ./build/tracker_tests
```

Sources (must be listed in `CMakeLists.txt`; not globbed):

| File | What it covers |
| --- | --- |
| `tests/sanity_test.cpp` | Eigen actually links (`Vector2d(3,4).norm() == 5`) |
| `tests/hungarian_test.cpp` | square optimum, greedy-fail case, wide/tall (unmatched → `-1`), empty dims, negative costs, ragged/NaN/inf rejected, MOT-style `1-IoU` reorder |
| `tests/multi_object_tracker_test.cpp` | empty update, ID stability under detection reorder, drop after `kMaximumMissedFrames` (5) empty frames, IoU gate `kMinimumIou` (0.30) refuses a far box and starts a new id |

Not covered in C++ tests: YOLO ONNX load/infer, MOG2, `VideoSource`, `evaluate_mot` I/O, Kalman numerics as a unit (Kalman is exercised only through `MultiObjectTracker`).

## Python (pytest)

`pyproject.toml`: `pythonpath = ["."]`. Run from repo root so `evaluation` imports.

```sh
uv sync
uv run pytest
uv run pytest tests/test_evaluation.py -q
```

`tests/test_evaluation.py` is offline (no download, no C++ build):

- `match_boxes` is one-to-one at IoU 0.5 (extra overlapping pred does not steal a second match)
- `suppress_ignored_predictions` keeps a pred that matches a pedestrian even if it also sits on an ignore box; drops it if there is no target
- `parse_mot(..., ground_truth=True)` reads `frame,id,x,y,w,h,conf,class,vis`
- `parse_mot` rejects `w==0`
- IoU of two 10×10 boxes offset by 5 in x is `1/3`

No pytest coverage of `calculate_metrics`, downloads, or `evaluate_mot` subprocess.

## End-to-end (not a unit test)

```sh
uv run python evaluation/evaluate.py
```

Needs `data/yolo11n.onnx`, network for the MOT17 subset (or a populated `data/mot17/`), and a successful preset debug build of `evaluate_mot`.
