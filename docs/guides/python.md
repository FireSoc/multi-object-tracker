# Python

There is no installable tracker package and no pybind/nanobind surface. `pyproject.toml` is a uv project (`name = "multi-object-tracker"`, `0.1.0`, `requires-python = ">=3.12"`) that pins tooling. `[tool.pytest.ini_options] pythonpath = ["."]` is the only package-path hook.

## Setup

```sh
uv sync
```

Do not create a venv by hand. `.venv/` is gitignored. Invoke everything with `uv run`.

Locked deps: `motmetrics>=1.4.0`, `numpy<2`, `onnx>=1.12,<2`, `onnxruntime>=1.29`, `onnxslim>=0.1.82`, `pytest>=8`, `ultralytics>=8.4.121`. `evaluation/evaluate.py` also imports `scipy.optimize.linear_sum_assignment` (transitive via motmetrics, not a direct dep).

## What you can run

YOLO export (Ultralytics CLI shipped by the `ultralytics` dep):

```sh
uv run yolo export model=data/yolo11n.pt format=onnx imgsz=640 opset=12 simplify=True
```

Eval driver:

```sh
uv run python evaluation/evaluate.py
uv run python evaluation/evaluate.py --frames 50 --iou 0.5
```

Tests:

```sh
uv run pytest
uv run pytest tests/test_evaluation.py
```

## Module: `evaluation`

`evaluation/__init__.py` is a one-line docstring (`"MOT17 evaluation utilities."`). Real code is `evaluation/evaluate.py`. Importable because pytest `pythonpath` includes the repo root:

```python
from evaluation.evaluate import (
    Row,
    intersection_over_union,
    match_boxes,
    parse_mot,
    suppress_ignored_predictions,
    calculate_metrics,
)
```

| Symbol | Role |
| --- | --- |
| `Row` | frozen dataclass: `frame`, `identity`, `box=(x,y,w,h)`, `confidence`, `class_id`, `visibility` |
| `intersection_over_union` | axis-aligned IoU on `(x,y,w,h)` |
| `match_boxes` | Hungarian one-to-one, keep pairs with IoU ≥ threshold |
| `suppress_ignored_predictions` | drop preds on ignore regions unless they match a pedestrian |
| `parse_mot` | MOT CSV → `{frame: list[Row]}` |
| `calculate_metrics` | precision/recall + motmetrics MOTA/IDF1/FP/FN/IDsw |
| `SEQUENCE` / `FRAME_COUNT` / `IOU_THRESHOLD` | `"MOT17-05-FRCNN"`, `100`, `0.50` |
| `PEDESTRIAN_CLASS` / `IGNORED_CLASSES` | `1`, `{2, 7, 8, 12}` |

`main()` is the CLI; not a library entry point beyond running the file. Constants like `MIRROR_COMMIT` are not CLI-overridable.

C++ types (`MultiObjectTracker`, `YoloDetector`, …) are not exposed to Python.
