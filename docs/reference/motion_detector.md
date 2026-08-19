# `motion_detector.hpp`

MOG2 background-subtraction blob detector. Same output type as [`YoloDetector`](yolo_detector.md) so [`MultiObjectTracker`](multi_object_tracker.md) does not care which detector you use.

```cpp
#include "motion_detector.hpp"
```

Implementation: `src/motion_detector.cpp`. Header includes `<opencv2/opencv.hpp>` (umbrella). The `.cpp` uses `imgproc` + `geometry` (`contourArea`, `boundingRect`, morphology, `findContours`).

## `MotionDetector`

```cpp
struct MotionDetector {
    cv::Ptr<cv::BackgroundSubtractor> subtractor = cv::createBackgroundSubtractorMOG2();
    double min_area = 500.0;
    std::vector<DetectedObject> detect(const cv::Mat& frame);
};
```

All members are **public**. Default construction is enough (`apps/main.cpp` uses `std::make_unique<MotionDetector>()`).

This is a **stateful** detector: `subtractor` accumulates a background model across `detect` calls. Do not reuse one instance across unrelated videos without replacing `subtractor`.

---

### `subtractor`

```cpp
cv::Ptr<cv::BackgroundSubtractor> subtractor = cv::createBackgroundSubtractorMOG2();
```

Default is OpenCV MOG2 with **library defaults**: `history=500`, `varThreshold=16`, `detectShadows=true`. Shadows appear as `127` in the mask and are dropped later.

You may assign another subtractor (e.g. `createBackgroundSubtractorKNN()`) before `detect`. There is no setter; just write the member. `detect` only requires `apply(frame, mask)`.

`cv::Ptr` — do not let this dangle; it owns the algorithm.

---

### `min_area`

```cpp
double min_area = 500.0;
```

Minimum `cv::contourArea` in **pixels²** to emit a detection. Default `500`. Contours below this are skipped. This is contour area, not bounding-box area.

---

### `detect`

```cpp
std::vector<DetectedObject> detect(const cv::Mat& frame);
```

**Purpose.** Update the background model, extract external contours, emit boxes.

**Parameters.**

| Name | Role |
| --- | --- |
| `frame` | Image passed to `BackgroundSubtractor::apply`. Typically BGR `CV_8UC3` from [`VideoSource`](video_source.md). |

**Returns.** [`DetectedObject`](detection.md) per surviving contour, in `findContours` order (OpenCV does not guarantee left-to-right).

- `bounding_box` — `cv::boundingRect(contour)`, integer pixels.
- `area_confidence` — `float(contourArea / (frame.rows * frame.cols))`, i.e. **fraction of the frame**, not a classifier score. Can exceed the YOLO-style `[0,1]` interpretation only if area exceeds the frame (it should not). If `frame.rows * frame.cols == 0`, this divides by zero.

**Error behavior.** **No checks.** Unlike YOLO, an empty `frame` is not rejected. `apply` on empty input is OpenCV-defined (often throws `cv::Exception` or produces an empty mask). This function does not throw on its own.

**Preconditions.** `subtractor` non-null (true after default init). `frame` should be non-empty, same size over time for a stable model (MOG2 will re-adapt if size changes, but you will get a burst of foreground).

**Postconditions.** Background model updated (`learningRate` default `-1` inside `apply`). `frame` is not modified.

---

## Implementation (`src/motion_detector.cpp`)

Pipeline, in order:

1. **Foreground mask.** `subtractor->apply(frame, mask)`  
   Values: `0` background, `127` shadow (MOG2), `255` foreground.

2. **Drop shadows.** `cv::threshold(mask, mask, 200, 255, THRESH_BINARY)`  
   Keeps only pixels `> 200`, so `127` shadows become `0`.

3. **Morphology.** Ellipse kernel `5×5`:
   - `MORPH_OPEN` (denoise)
   - `MORPH_CLOSE` (fill holes)

4. **Contours.** `findContours(mask, …, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`  
   External contours only; nested holes ignored.

5. **Filter + pack.** Skip `contourArea < min_area`. Else `{boundingRect, area/frameArea}`.

**Complexity.** Dominated by MOG2 `apply` (`O(pixels × GMM components)`) plus `findContours`. Morphology is 5×5 ellipse, two passes.

**Memory.** Reallocates `mask`, `kernel`, `contours` every call. The subtractor’s GMM state is the long-lived allocation.

**Threading.** Not safe. `apply` mutates the model.

**Cold start.** The first tens–hundreds of frames typically light up large foreground regions until the background estimate settles. There is no “learning-only” flag on this API (`apply`’s optional `learningRate` is not exposed; always default).

**Not done.** Connected-component stats instead of contours, shadow-as-class, ROI, downscaling before MOG2, returning [`ClassifiedObject`](detection.md#classifiedobject).

## Typical use

```cpp
MotionDetector motion;
VideoSource source(0);
cv::Mat frame;
MultiObjectTracker tracker;
while (source.next(frame)) {
    tracker.update(motion.detect(frame));
}
```

`apps/main.cpp` uses this when `--yolo` is not passed.

## See also

- [`YoloDetector`](yolo_detector.md) — alternative `vector<DetectedObject>` source
- [`DetectedObject`](detection.md#detectedobject)
- OpenCV `cv::BackgroundSubtractorMOG2`
