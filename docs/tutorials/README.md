# Tutorials

Learning-oriented. A tutorial is a supervised first success, not a command reference. Keep explanation to a sentence and link to [explanation](../explanation/README.md) or [reference](../reference/README.md).

Audience: C++/OpenCV fluent, new to *this* tree.

## Pages

| Page | What the reader will have done |
|---|---|
| [first-run.md](first-run.md) | Built `tracker`, opened camera 0 (or a file), seen MOG2 boxes with `ID N`, quit with `q`. Optional `--yolo` once an ONNX exists. |

Do not put `ctest`, MOT17 download, or `evaluate_mot` argv here — those are [guides](../guides/README.md).

## Code this tutorial actually runs

- `cmake` + `tracker` target (`apps/main.cpp`)
- Default path: `MotionDetector` (`include/motion_detector.hpp`)
- `--yolo` path: `YoloDetector` (`include/yolo_detector.hpp`)
- Overlay loop: `MultiObjectTracker::update` → `TrackedObject::{id, bounding_box}`

## See also

- [How-to](../how-to/README.md) / [guides](../guides/README.md) — export, eval, `ctest`
- [Overview](../overview.md) — component map
- [Reference](../reference/README.md) — types and CLI
- [Explanation](../explanation/README.md) — why the pipeline is this shape
