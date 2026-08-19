# How-to

Goal-oriented procedures. Content lives in [guides/](../guides/README.md), not in this folder.

The reader already compiled something or knows they want a specific outcome. Numbered steps, no SORT lecture.

## Pages (in `docs/guides/`)

| Goal | Page |
|---|---|
| Configure, build, both output dirs (`-B build` vs `--preset debug`) | [guides/build.md](../guides/build.md) |
| Run GoogleTest / pytest | [guides/testing.md](../guides/testing.md) |
| `uv sync` + Ultralytics ONNX export | [guides/python.md](../guides/python.md) |
| All four `tracker` invocations + `evaluate_mot` argv | [guides/run.md](../guides/run.md) |
| Reproduce the MOT17-05-FRCNN smoke run | [guides/evaluation.md](../guides/evaluation.md) |

## Facts that must stay in reference

CLI syntax, `YoloDetector` thresholds, MOT CSV columns, CMake target list: [reference](../reference/README.md). “Why ignore classes 2/7/8/12”: [explanation](../explanation/README.md).

## See also

- [Guides index](../guides/README.md)
- [Tutorials](../tutorials/README.md) if they have never run `tracker`
- [Overview](../overview.md) for artifact names
- [Reference](../reference/README.md)
