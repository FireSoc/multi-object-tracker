# Build

CMake 3.24+, C++20, no GNU extensions. Default `CMAKE_BUILD_TYPE` is `Debug` if you do not set one. Compile flags: `-Wall -Wextra -Wpedantic`. `CMAKE_EXPORT_COMPILE_COMMANDS` is on; `.clangd` points `CompilationDatabase` at `build/debug`.

## System deps (C++)

`find_package` — not vendored, not installed by `uv`:

- OpenCV components: `core`, `imgproc`, `imgcodecs`, `videoio`, `highgui`, `dnn`, `video`, `geometry`
- Eigen3 (`NO_MODULE`)

`uv` only owns Python tooling (ONNX export, motmetrics, pytest). OpenCV DNN is what actually runs YOLO at inference time.

## Layout

`tracker_core` is the library (`src/` + `include/`). Apps and tests link it. Examples do not: each `examples/*.cpp` becomes its own exe linked only to OpenCV + Eigen.

| Target | Source | Notes |
| --- | --- | --- |
| `tracker_core` | `src/{video_source,motion_detector,yolo_detector,hungarian,multi_object_tracker}.cpp` | no `main` |
| `tracker` | `apps/main.cpp` | interactive |
| `evaluate_mot` | `apps/evaluate_mot.cpp` | headless MOT dump |
| `tracker_tests` | `tests/{sanity,hungarian,multi_object_tracker}_test.cpp` | GoogleTest, `BUILD_TESTS=ON` (default) |
| `shapes`, `img_transform`, `opencv_prac` | `examples/*.cpp` | glob; each file needs its own `main` |

GoogleTest v1.15.2 is FetchContent'd from GitHub on first configure when tests are on. Needs network that once.

## Two output directories

**Presets** (`CMakePresets.json`) — this is what `evaluation/evaluate.py` and clangd use:

```sh
cmake --preset debug
cmake --build --preset debug
```

Binaries land in `build/debug/` (`tracker`, `evaluate_mot`, `tracker_tests`, example names). Release:

```sh
cmake --preset release
cmake --build --preset release
```

→ `build/release/`.

**Ad-hoc** (README):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

→ `build/`. `evaluate.py` will not find `build/debug/evaluate_mot` if you only did this.

Build one target:

```sh
cmake --build --preset debug --target evaluate_mot
cmake --build build --target tracker
```

Skip tests (no gtest download):

```sh
cmake --preset debug -DBUILD_TESTS=OFF
```

`build/` and `cmake-build-*/` are gitignored.

## Python (uv)

Requires Python ≥ 3.12. From repo root:

```sh
uv sync
```

Creates `.venv/` (gitignored) from `pyproject.toml` + `uv.lock`. Direct deps: `motmetrics`, `numpy<2`, `onnx`, `onnxruntime`, `onnxslim`, `pytest`, `ultralytics`. There is no `[build-system]` and no C++ extension; this is a dependency lock, not a pip-installable tracker package.

See [python.md](python.md).
