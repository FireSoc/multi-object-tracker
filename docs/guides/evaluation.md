# Evaluation

Smoke eval on frames 1–100 of MOT17 train `MOT17-05-FRCNN` (640×480). Not a MOT17 benchmark: generic COCO YOLO11n, no MOT-tuned detector, SORT with IoU 0.30 / 5-frame coast.

## One command

From repo root, after `uv sync` and with `data/yolo11n.onnx` present:

```sh
uv run python evaluation/evaluate.py
```

Optional (argparse; `--help` is the generated one):

| Flag | Default | Constraint |
| --- | --- | --- |
| `--frames` | `100` | integer in `[1, 100]` |
| `--iou` | `0.50` | float in `(0, 1]` |

No `--model`, sequence, or output-dir flags. Model is always `data/yolo11n.onnx`. Missing model → `FileNotFoundError`.

What it does:

1. Download (skip if present) `gt/gt.txt` and `img1/000001.jpg`…`{frames}.jpg` from Hugging Face `Lekim89/MOT17` at commit `f93908467986c77765667925a526ad07be8ad630`. Cap 500 MiB. UA `mot-evaluation/1.0`. Writes `data/mot17/` (gitignored).
2. `cmake --preset debug` then `cmake --build --preset debug --target evaluate_mot`.
3. Run `build/debug/evaluate_mot` → `data/mot17/results/{detections,tracks}.txt`.
4. Score and print.

## File format

MOT Challenge CSV, comma-separated, no header. Parser (`parse_mot`) requires ≥7 columns; non-numeric / `frame<=0` / `w<=0` / `h<=0` raise `ValueError`.

```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>[, <class>, <visibility>[, ...]]
```

Ground truth (`gt.txt`, `ground_truth=True`): columns 8–9 are class and visibility. Predictions: class/visibility ignored (stored `-1`). `evaluate_mot` writes ten columns, last three `-1`. Boxes from the C++ writer are 1-indexed (`x+1`, `y+1`).

## How scores are computed

All matching is one-to-one Hungarian (`scipy.optimize.linear_sum_assignment`) on cost `1 - IoU`, accepted iff IoU ≥ `--iou` (0.50 by default). This eval IoU is **not** the tracker’s association gate (0.30).

Per frame, GT kept only if `conf == 1.0` (exact float). Split:

- **targets**: MOT class `1` (pedestrian)
- **ignored**: classes `{2, 7, 8, 12}` (person-on-vehicle, static person, distractor, reflection)
- other classes dropped

No visibility cutoff. `visibility` is parsed and unused.

Ignore handling (`suppress_ignored_predictions`): a prediction overlapping an ignore box at IoU ≥ threshold is removed **unless** it was matched to a pedestrian first (protected). Same pass on detections and tracks.

**Detection precision / recall** (not motmetrics): after ignore suppression, match detections to pedestrians. `P = TP/(TP+FP)`, `R = TP/(TP+FN)`, zeros if the denominator is 0.

**Tracking MOTA / IDF1 / FP / FN / ID switches**: `motmetrics.MOTAccumulator(auto_id=False)`. Distance is `1 - IoU` if IoU ≥ threshold else `NaN`. Object ids are GT identities vs tracker ids. Metrics requested: `mota`, `idf1`, `num_false_positives`, `num_misses`, `num_switches`. Printed as `fp` / `fn` / `id_switches`.

Stdout:

```
Downloaded <MiB>
MOT17-05-FRCNN frames 1-<N>, IoU 0.50
precision: ...
recall: ...
mota: ...
idf1: ...
fp: ...
fn: ...
id_switches: ...
```

Floats `:.4f`, counts unformatted. README’s 100-frame numbers (P 0.7217, R 0.4706, MOTA 0.1151, IDF1 0.4680, FP 330, FN 364, IDs 13) are that integration baseline, not a claim across MOT17.

## Calling `evaluate_mot` yourself

```
evaluate_mot <model.onnx> <image-dir> <detections.csv> <tracks.csv> <frame-count>
```

Same writer as the script. You still need this Python scorer (or motmetrics) on the two CSVs plus `gt.txt`. Detector in that binary: person-only, conf 0.25, NMS 0.45, 640². Tracker miss window 5 is included in `tracks.txt`.
