# Guides

How to build, run, evaluate, and test this repo. Commands and flags are taken from `CMakeLists.txt`, `CMakePresets.json`, `apps/`, `evaluation/evaluate.py`, and `pyproject.toml`. Nothing here is a public Python API wrap of the C++ tracker.

This folder is the how-to quadrant. The Diátaxis pointer is [how-to/README.md](../how-to/README.md).

| Guide | What it covers |
| --- | --- |
| [build.md](build.md) | CMake, OpenCV/Eigen, uv/Python, compile layouts |
| [run.md](run.md) | `tracker`, `evaluate_mot`, examples, models, I/O |
| [evaluation.md](evaluation.md) | MOT17 subset, metrics, CSV format, scoring |
| [python.md](python.md) | `uv` deps, `evaluation` module, YOLO export |
| [testing.md](testing.md) | GoogleTest + pytest, what each file covers |

Two CMake layouts exist. README uses `-B build`. Presets and `evaluation/evaluate.py` use `build/debug`. Use one and stick to it, or the binaries will not be where you expect.

## See also

- [Overview](../overview.md) — what ships and which binary is which
- [Tutorials](../tutorials/README.md) — first `tracker` run
- [Reference](../reference/README.md) — APIs, CLI synopsis, MOT columns
- [Explanation](../explanation/README.md) — why eval ignores MOT classes 2/7/8/12
