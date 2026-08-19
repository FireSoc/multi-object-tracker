# Documentation conventions

How this repo’s docs are written and where a page belongs. Grounded in Diátaxis and in how Kubernetes, PyTorch, NumPy, OpenCV, Rust, and Ultralytics actually ship docs — adapted to a small C++ library with a Python evaluation sidecar.

Read this before adding a page. [README.md](README.md) is the reader-facing map; this file is the author-facing contract.

## Framework

[Diátaxis](https://diataxis.fr/) splits docs by *user need*, not by module. Keep that as the conceptual model. Physical folders follow where the pages actually live (`guides/`, `architecture/`, `algorithms/`); `how-to/` and `explanation/` are indexes, not content dumps.

| Quadrant | Need | Index | Content lives in | Kubernetes analogue | PyTorch analogue |
|---|---|---|---|---|---|
| Tutorials | acquire skill by doing | `docs/tutorials/` | `docs/tutorials/` | Tutorials | `tutorials.pytorch.org` |
| How-to | apply skill to a real goal | `docs/how-to/README.md` | `docs/guides/` | Tasks | Recipes |
| Reference | look up facts | `docs/reference/` | `docs/reference/` | Reference | `docs.pytorch.org` |
| Explanation | understand why | `docs/explanation/README.md` | `docs/architecture/` + `docs/algorithms/` | Concepts | notes / design docs |

Axes: action vs cognition, study vs work. Tutorials and how-tos are procedures. Reference and explanation are not.

Production projects that keep this split (and that we copy):

- **Kubernetes** — explicit mapping; concept pages have no step lists; task pages have numbered steps and link out for theory; each section has an index. Page templates: tutorial (`prerequisites`, `objectives`, `cleanup`), task (`prerequisites`, numbered `steps`), concept (`overview`, body, ≤5 “what’s next”), reference (`synopsis`, `options`, `seealso`).
- **PyTorch** — long learning paths stay in Tutorials; “I need to do X” stays in Recipes; signatures stay in the API. Do not put `MultiObjectTracker::update` contract in a webcam walkthrough.
- **NumPy** — User Guide vs Reference. Their historical mix of how-to and explanation in one “user guide” is what we *avoid* by using four *types*. We do not force Django-style folder names when the content already lives under `guides/` / `architecture/` / `algorithms/`.
- **OpenCV** — `tutorials/` for first success; Doxygen for `cv::KalmanFilter` / `cv::dnn::Net` facts. Our C++ types get Markdown reference pages (no Doxygen site yet); still treat them as reference, not tutorials.
- **Rust** — The Book interleaves tutorial and explanation; we do **not** copy that for a library this size. `std` docs are the reference model: types, invariants, no narrative.
- **Ultralytics** — `docs/en/modes/track.md` is a hybrid (CLI/Python how-to + tracker comparison table + paper links). Useful as a *topic hub*, but we split: run commands → `guides/`; `kMinimumIou` / CLI → `reference/`; SORT vs ByteTrack → `algorithms/`.
- **SORT / ByteTrack accompanying material** — papers are explanation; READMEs are how-to plus result tables; MOTChallenge format is reference. Reproduce that split: MOTA numbers and ignore-class policy do not belong in `tutorials/`.

## Folder rules

One primary type per page. If the reader needs another type, link.

| Put it in… | If it is… | Not if it is… |
|---|---|---|
| `tutorials/` | A guided first success the author has tested end-to-end | A command card for someone who already built the tree |
| `guides/` | “Export YOLO to ONNX”, “score MOT17-05 frames 1–100”, “run `ctest`” | A walkthrough that teaches OpenCV from zero |
| `reference/` | Signatures, CLI, constants, MOT CSV columns, CMake targets, ONNX layout assumptions | Why the Kalman state is 6-D |
| `architecture/` | System shape, ownership, types at boundaries | Kalman predict equations |
| `algorithms/` | Association policy, motion model, detector internals, track lifecycle | `HungarianAlgorithm::solve` return-value contract |

Top-level exceptions (owned, not duplicated into a quadrant):

- `docs/README.md` — navigation hub (Kubernetes docs home).
- `docs/CONVENTIONS.md` — this file.
- `docs/overview.md` — system map (components, languages, artifacts). Deeper “why” goes under `architecture/` / `algorithms/` (indexed from `explanation/`).
- `docs/how-to/README.md` — pointer into `guides/`.
- `docs/explanation/README.md` — pointer into `architecture/` + `algorithms/`.

`examples/*.cpp` are OpenCV practice binaries (`CMakeLists.txt` GLOB), not tracker tutorials. Do not document them as the tracking API.

## Naming

- Indexes: `README.md` (GitHub renders them; matches the repo README).
- Tutorials, guides, architecture, algorithms: `kebab-case.md`.
- Reference pages match the C++ header stem: `reference/hungarian.md` ← `include/hungarian.hpp`, not `hungarian-algorithm.md`.
- One topic per file. Prefer `reference/yolo_detector.md` over `reference/detectors.md`.
- Headings: sentence case (`## Track lifecycle`, not `## Track Lifecycle`).
- Title (`#`) states the object or goal, not the Diátaxis type. The folder already is the type.
- Symbols in prose: match the code. `` `MultiObjectTracker::update` ``, `` `kMaximumMissedFrames` ``, `` `evaluate_mot` ``.

## Linking

Relative Markdown only for in-repo targets.

From a page under `docs/` (this file’s directory):

```markdown
[Track association](algorithms/association.md)
[`MultiObjectTracker`](reference/multi_object_tracker.md)
[`include/multi_object_tracker.hpp`](../include/multi_object_tracker.hpp)
```

From a nested page (`docs/guides/`, `docs/algorithms/`, …) prefix with `../`.

Rules:

- Cite real paths and identifiers. If it is not in `include/`, `src/`, `apps/`, `evaluation/`, or `CMakeLists.txt`, do not invent it.
- Cross-quadrant links at the end of the page, Kubernetes-style: a short **What’s next** / **See also** list, **at most five** bullets.
- Area indexes (`architecture/`, `algorithms/`, `reference/`) cross-link the *same types* (detection, association, Kalman, lifecycle). Guides cross-link [overview.md](overview.md).
- Do not deep-link a how-to from a reference page as if it were the API. Point at the reference page for facts.
- External links are for papers, MOTChallenge, Ultralytics export docs, OpenCV modules — not for mirroring this tree.
- The repo [README.md](../README.md) points here once. Do not fork a second architecture write-up there.

## Page skeletons

Copy the Kubernetes section intent, not their Hugo shortcodes.

**Tutorial** — learning-oriented. Minimize theory (Diátaxis: “ruthlessly minimise explanation”).

1. Goal the reader will have completed.
2. Prerequisites (CMake, OpenCV, a camera or a file).
3. Numbered steps that must work.
4. What they should see (`ID N` overlays, window title `tracker`).
5. Cleanup / quit (`q` in `apps/main.cpp`).
6. What’s next → `guides/` or `reference/`.

**How-to** (`guides/`) — competent reader, one goal.

1. One-paragraph motivation so they can bail if it is the wrong page.
2. Prerequisites.
3. Numbered commands. Copy-pasteable. Match real binaries: `./build/tracker` vs `build/debug/evaluate_mot` depending on whether they used `-B build` or `--preset debug`.
4. No architecture essay. Link `overview.md` or `explanation/`.

**Reference** — propositional. Read like `std::vector` docs.

1. Synopsis (type / target / CLI).
2. Header / source paths.
3. Parameters, invariants, error cases (`std::invalid_argument` vs `std::runtime_error` as the code actually throws).
4. Related types. No “now try this” except a see-also how-to.

**Explanation** (`architecture/` + `algorithms/`) — why.

1. Claim.
2. What the code does (cite functions).
3. What it deliberately is not (e.g. not ByteTrack two-stage association).
4. Links to the reference page for constants.

## What belongs in each quadrant (this codebase)

Use this as the assignment table. Do not collapse these into `overview.md`.

| Topic | Folder | Notes |
|---|---|---|
| Webcam + MOG2 first run | `tutorials/first-run.md` | `MotionDetector` default path in `apps/main.cpp` |
| YOLO ONNX first run | `tutorials/first-run.md` | `--yolo <model.onnx>`; model not bundled |
| Configure + build | `guides/build.md` | `CMakeLists.txt`, `CMakePresets.json` |
| GoogleTest / pytest | `guides/testing.md` | `ctest`, `uv run pytest` |
| `uv sync` + `yolo export` | `guides/python.md` | `pyproject.toml`; uv does **not** install OpenCV/Eigen |
| CLI recipes for `tracker` / `evaluate_mot` | `guides/run.md` | four `tracker` invocations; `evaluate_mot` argv |
| MOT17-05 smoke eval | `guides/evaluation.md` | `uv run python evaluation/evaluate.py` |
| `DetectedObject` / `ClassifiedObject` | `reference/detection.md` | `ClassifiedObject` is currently unused |
| `VideoSource` | `reference/video_source.md` | `include/video_source.hpp` |
| `MotionDetector` | `reference/motion_detector.md` | MOG2, `min_area = 500`, threshold 200 |
| `YoloDetector` | `reference/yolo_detector.md` | 640 input, conf 0.25, NMS 0.45, ONNX via `cv::dnn::readNetFromONNX` |
| `BoundingBoxKalmanFilter` | `reference/kalman.md` | OpenCV `cv::KalmanFilter`, 6-state / 4-measure — **not** Eigen, despite `find_package(Eigen3)` |
| `HungarianAlgorithm::solve` | `reference/hungarian.md` | rectangular costs, `-1` unmatched rows, finite costs only |
| `MultiObjectTracker` / `TrackedObject` | `reference/multi_object_tracker.md` | `kMinimumIou = 0.30`, `kMaximumMissedFrames = 5`, IDs from 1 |
| `tracker` / `evaluate_mot` CLI | `reference/README.md` | exact usage strings in `apps/`; recipes in `guides/run.md` |
| MOT CSV + metrics names | `reference/README.md` + `guides/evaluation.md` | `write_box` 1-based `x,y`; `motmetrics` keys in `calculate_metrics` |
| CMake targets | `reference/README.md` + `guides/build.md` | `tracker_core`, `tracker`, `evaluate_mot`, example GLOB, `tracker_tests` |
| SORT pipeline | `architecture/pipeline.md`, `algorithms/` | `src/multi_object_tracker.cpp` |
| Detector choice (MOG2 vs YOLO) | `algorithms/detection.md` | mutually exclusive in `apps/main.cpp`; tracker is detector-agnostic |
| Evaluation protocol (ignore classes, no visibility cutoff) | `architecture/data-flow.md`, `guides/evaluation.md` | `evaluation/evaluate.py` |

## Tone and accuracy

- Expert audience. No MOT 101. No “in this article we will…”.
- If the code and a comment disagree, document the code and note the comment.
- Known mismatches that docs must not paper over:
  - `CMakeLists.txt` says Eigen is “for the Kalman filter”; `include/kalman.hpp` uses `cv::KalmanFilter`. Eigen is linked and sanity-checked in `tests/sanity_test.cpp` only.
  - Repo README `cmake -S . -B build` → `./build/tracker`. Presets → `build/debug/` and `build/release/`. Evaluation uses presets.
- Do not document planned APIs. `ClassifiedObject` exists; do not describe a classification pipeline.
- Do not modify C++/Python source from a docs change.

## Index pages

Diátaxis indexes:

- `tutorials/README.md`, `how-to/README.md`, `reference/README.md`, `explanation/README.md`

Physical-folder indexes (also required):

- `guides/README.md`, `architecture/README.md`, `algorithms/README.md`

Each index:

- States the need in one paragraph.
- Lists pages with a one-line description and the code they cover.
- Does not contain the procedure or the API itself (`reference/README.md` may hold short CLI / MOT / CMake synopses that would otherwise be stub pages).
- Has a **See also** to sibling areas.

When you add a page, link it from that folder’s `README.md` and from the Diátaxis index if the page is a how-to or explanation. Link from [docs/README.md](README.md) only if the tree map needs a new *folder* — not every leaf.
