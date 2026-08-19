# `video_source.hpp`

Thin RAII wrapper around `cv::VideoCapture` for webcam or file.

```cpp
#include "video_source.hpp"
```

Implementation: `src/video_source.cpp`. Depends on OpenCV `core` + `videoio`.

## `VideoSource`

```cpp
struct VideoSource {
    explicit VideoSource(int camera_index = 0);
    explicit VideoSource(const std::string& video_path);

    bool is_open() const;
    bool next(cv::Mat& frame);

    cv::VideoCapture capture_;
};
```

A **struct** with a **public** `capture_`. That is intentional: callers can set `cv::CAP_PROP_*` without extra accessors. There is no destructor beyond the implicit one (which releases the capture).

The two constructors are `explicit`; there is no default constructor other than `VideoSource()` via the defaulted `camera_index = 0`.

Not copy-oriented in the apps: `main` constructs one source and loops `next`.

---

### `VideoSource(int camera_index = 0)`

```cpp
explicit VideoSource(int camera_index = 0);
```

```cpp
: capture_(camera_index) {}
```

Opens device `camera_index` with OpenCV’s default backend (`cv::VideoCapture::VideoCapture(int)`). Index `0` is the default camera (`apps/main.cpp` uses `kDefaultCameraIndex = 0`).

**Does not throw** if the device is missing. Check [`is_open`](#is_open).

---

### `VideoSource(const std::string& video_path)`

```cpp
explicit VideoSource(const std::string& video_path);
```

```cpp
: capture_(video_path) {}
```

Opens a file (or URL, if the OpenCV build supports it) via `cv::VideoCapture::VideoCapture(const String&)`.

Empty path: OpenCV typically fails to open; again, [`is_open`](#is_open) is the check. `main` rejects empty/`--` argv before constructing.

---

### `is_open`

```cpp
bool is_open() const;
```

Returns `capture_.isOpened()`.

**Typical use.**

```cpp
VideoSource source = using_video_file ? VideoSource(path) : VideoSource(0);
if (!source.is_open()) { /* fail */ }
```

---

### `next`

```cpp
bool next(cv::Mat& frame);
```

**Purpose.** Grab the next frame into `frame`.

**Parameters.**

| Name | Role |
| --- | --- |
| `frame` | Output. Overwritten by `capture_.read(frame)`. On failure OpenCV empties it. |

**Returns.** `true` if a frame was decoded; `false` on EOF, disconnected camera, or closed capture.

**Preconditions.** None enforced. Calling `next` on a closed source returns `false`.

**Postconditions.** On `true`, `frame` is a BGR `CV_8UC3` image for typical files/webcams (whatever the capture produces — this wrapper does not convert color). On `false`, treat `frame` as empty.

**Blocking.** `read` blocks until a camera frame arrives. File reads are sequential; there is no seeking API on `VideoSource` itself (`capture_.set(cv::CAP_PROP_POS_FRAMES, …)` works because `capture_` is public).

**Complexity.** Dominated by decode. No extra copies in this wrapper.

---

### `capture_`

Public `cv::VideoCapture`. Lifetime is the `VideoSource`. Do not keep a reference to `capture_` after the `VideoSource` dies.

Useful knobs (not wrapped):

```cpp
source.capture_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
source.capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
source.capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
```

---

## Typical use

```cpp
VideoSource source(argc > 1 ? VideoSource(argv[1]) : VideoSource(0));
cv::Mat frame;
while (source.next(frame)) {
    auto detections = detector.detect(frame);
    tracker.update(detections);
}
```

`evaluate_mot` does **not** use `VideoSource`; it `cv::imread`s numbered JPEGs.

## Threading / errors

`cv::VideoCapture` is not thread-safe. This type does not throw; OpenCV may throw `cv::Exception` from `read` on some backends — `main` catches that.

No internal buffering beyond OpenCV’s capture buffer.

## See also

- [`YoloDetector::detect`](yolo_detector.md#detect) / [`MotionDetector::detect`](motion_detector.md#detect) — consumers of `frame`
