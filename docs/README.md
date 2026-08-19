# Documentation

C++20 SORT-style multi-object tracker: OpenCV detections in, persistent track IDs out. These docs are for someone who already knows Kalman filters, IoU, and MOTChallenge metrics, and needs the *this-repo* map.

Repo README ([README.md](../README.md)) stays the short command card. Everything that needs a type (tutorial vs how-to vs reference vs explanation) lives here.

## How to navigate

Diátaxis by *why you opened the docs*. Physical folders keep the names the content agents used (`guides/`, `architecture/`, `algorithms/`); the indexes below are the map.

| You want to… | Diátaxis | Open |
|---|---|---|
| Get a first successful run under guidance | Tutorials | [tutorials/](tutorials/README.md) |
| Perform a known task (build, export ONNX, evaluate MOT17) | How-to | [guides/](guides/README.md) ([how-to index](how-to/README.md)) |
| Look up an API, CLI, constant, file format, or CMake target | Reference | [reference/](reference/README.md) |
| Understand *why* the pipeline is shaped this way | Explanation | [architecture/](architecture/README.md) and [algorithms/](algorithms/README.md) ([explanation index](explanation/README.md)) |

Project-level context (what ships, what talks to what) is [overview.md](overview.md). How to write and where to put new pages is [CONVENTIONS.md](CONVENTIONS.md).

Do not dump MOT theory into a tutorial, or `tracker` CLI flags into an architecture page. Link across folders instead.

## Audience

| Reader | Start |
|---|---|
| Engineer wiring `MultiObjectTracker` into another binary | [overview.md](overview.md) → [reference/](reference/README.md) |
| Someone reproducing the MOT17-05 smoke numbers | [guides/evaluation.md](guides/evaluation.md) |
| Someone extending association / motion / detectors | [explanation/](explanation/README.md) then the matching [reference](reference/README.md) page |
| First checkout, wants pixels on screen | [tutorials/first-run.md](tutorials/first-run.md) |

Python here is not a tracker runtime. It is Ultralytics export (`uv run yolo export …`) plus the MOT scorer in `evaluation/evaluate.py`. The track loop is C++.

## Docs tree

```
docs/
  README.md                 this page
  CONVENTIONS.md            authoring rules
  overview.md               what ships: components, languages, artifacts
  tutorials/                first-success walkthrough
  how-to/README.md          pointer → guides/
  guides/                   how-to: build, run, eval, python, testing
  reference/                C++ APIs (one page per header) + CLI/format index
  explanation/README.md     pointer → architecture/ + algorithms/
  architecture/             system shape, data flow, components
  algorithms/               SORT stages as implemented
```

Code the docs must stay honest to:

- Library: `tracker_core` from `src/` + `include/` (`CMakeLists.txt`)
- Interactive binary: `tracker` ← `apps/main.cpp`
- Headless MOT writer: `evaluate_mot` ← `apps/evaluate_mot.cpp`
- Scorer: `evaluation/evaluate.py` (invokes the debug `evaluate_mot` preset build)
- Tests: `tests/*.cpp` (GoogleTest) and `tests/test_evaluation.py` (pytest)

## Sources

Conventions follow [Diátaxis](https://diataxis.fr/), Kubernetes page types (tutorial / task / concept / reference), and the PyTorch split (tutorials vs recipes vs API). Tracker-specific pages should cite this tree’s headers, not Ultralytics YAML tracker backends or the SORT/ByteTrack papers, unless an algorithms page is explicitly comparing implementations.
