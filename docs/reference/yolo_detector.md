# `yolo_detector.hpp`

Ultralytics YOLO (v8 / YOLO11 detection) via OpenCV DNN + ONNX. Outputs [`DetectedObject`](detection.md) in original-frame pixels.

```cpp
#include "yolo_detector.hpp"
```

Implementation: `src/yolo_detector.cpp`. Depends on OpenCV `core`, `dnn`, `imgproc`; also `<optional>`, `<string>`, `<vector>`.

## `YoloDetector`

```cpp
class YoloDetector {
   public:
    static constexpr int kInputSize = 640;
    static constexpr float kDefaultConfidenceThreshold = 0.25F;
    static constexpr float kDefaultNmsThreshold = 0.45F;

    explicit YoloDetector(const std::string& model_path,
                          float confidence_threshold = kDefaultConfidenceThreshold,
                          float nms_threshold = kDefaultNmsThreshold,
                          std::optional<int> class_id = std::nullopt);

    std::vector<DetectedObject> detect(const cv::Mat& frame);

   private:
    cv::dnn::Net network_;
    float confidence_threshold_;
    float nms_threshold_;
    std::optional<int> class_id_;
};
```

One loaded network per instance. Backend/target are OpenCV defaults (typically CPU). There is no API to set `DNN_BACKEND_CUDA` / `DNN_TARGET_CUDA`.

### Public constants

| Constant | Value | Role |
| --- | --- | --- |
| `kInputSize` | `640` | Blob spatial size. Must match the exported `imgsz`. |
| `kDefaultConfidenceThreshold` | `0.25F` | Per-candidate class-score cutoff |
| `kDefaultNmsThreshold` | `0.45F` | IoU threshold for `cv::dnn::NMSBoxes` |

`evaluate_mot` and the README smoke eval use these defaults plus `class_id = 0` (COCO person).

---

### Constructor

```cpp
explicit YoloDetector(const std::string& model_path,
                      float confidence_threshold = kDefaultConfidenceThreshold,
                      float nms_threshold = kDefaultNmsThreshold,
                      std::optional<int> class_id = std::nullopt);
```

**Purpose.** Load an ONNX graph and store thresholds / optional class filter.

**Parameters.**

| Name | Role |
| --- | --- |
| `model_path` | Filesystem path to an Ultralytics detection ONNX. Not bundled. |
| `confidence_threshold` | In `[0, 1]`. Candidates with selected class score `<` this are dropped **before** NMS. Also passed to `NMSBoxes` as `score_threshold`. |
| `nms_threshold` | In `[0, 1]`. IoU above which the lower-scoring box is suppressed. |
| `class_id` | If set, use `class_scores[class_id]` instead of `max(class_scores)`. COCO person is `0`. `nullopt` keeps the argmax class (still **one** score per box; the class index is **not** stored on [`DetectedObject`](detection.md)). |

**Error behavior.**

| Condition | Exception | Message |
| --- | --- | --- |
| `model_path.empty()` | `std::invalid_argument` | `"YOLO model path must not be empty"` |
| confidence not in `[0, 1]` | `std::invalid_argument` | `"YOLO confidence threshold must be in [0, 1]"` |
| NMS not in `[0, 1]` | `std::invalid_argument` | `"YOLO NMS threshold must be in [0, 1]"` |
| `class_id` set and `< 0` | `std::invalid_argument` | `"YOLO class ID must be non-negative"` |
| `cv::dnn::readNetFromONNX` throws | `std::runtime_error` | `"Failed to load YOLO ONNX model '<path>': " + cv::Exception::what()` |
| `network_.empty()` after load | `std::runtime_error` | `"Failed to load YOLO ONNX model '<path>'"` |

`class_id` vs model class count is **not** checked at construction; it is checked on the first candidate row in `detect` (needs the output width).

**Postconditions.** `network_` holds the graph. Thresholds copied into members.

**Export expected by this code.** Ultralytics `yolo export … format=onnx imgsz=640 opset=12` (see repo README). Output tensor rank 3, batch 1, layout either `[1, features, candidates]` (native Ultralytics, e.g. `[1, 84, 8400]` for 80 classes) or `[1, candidates, features]`. Features = `[cx, cy, w, h, class0, …]` in **640-space**, **no separate objectness**. Models with extra objectness, segmentation masks, or end-to-end NMS already applied will not parse as intended.

---

### `detect`

```cpp
std::vector<DetectedObject> detect(const cv::Mat& frame);
```

**Purpose.** Preprocess → forward → decode → NMS → `DetectedObject` list.

**Parameters.**

| Name | Role |
| --- | --- |
| `frame` | Non-empty image. Color interpretation: `blobFromImage(..., swapRB=true)` treats it as **BGR** (OpenCV default) and feeds **RGB** to the net. |

**Returns.** Zero or more [`DetectedObject`](detection.md#detectedobject):

- `bounding_box` — integer `cv::Rect` in **original** `frame` pixels, clamped to `[0, cols] × [0, rows]`.
- `area_confidence` — the class score that survived the threshold (argmax or `class_id`), **not** an area fraction.

Order is NMS keep order (`cv::dnn::NMSBoxes` index list), not left-to-right.

**Error behavior.**

| Condition | Exception |
| --- | --- |
| `frame.empty()` | `std::invalid_argument` `"Cannot run YOLO detection on an empty frame"` |
| Output rank ≠ 3, batch ≠ 1, or `channels() ≠ 1` | `std::runtime_error` `"YOLO output must have shape [1, features, candidates] or [1, candidates, features]"` |
| Decoded feature count `< 5` or candidate count `≤ 0` | `std::runtime_error` `"YOLO output has no class scores or candidates"` |
| `class_id` ≥ number of class scores | `std::runtime_error` `"YOLO class ID exceeds the model's class count"` |
| NMS returns an index outside `boxes` | `std::runtime_error` `"YOLO NMS returned an invalid candidate index"` |

OpenCV DNN failures surface as `cv::Exception`.

**Preconditions.** Constructed successfully. `frame.cols` and `frame.rows` used as scale denominators; both must be positive (empty already rejected).

**Postconditions.** No mutation of `frame`. Internal `network_` input is overwritten each call.

---

## Implementation (`src/yolo_detector.cpp`)

### Preprocess

```cpp
cv::dnn::blobFromImage(frame, blob,
                       /*scalefactor*/ 1.0/255.0,
                       cv::Size(kInputSize, kInputSize),
                       /*mean*/ cv::Scalar(),
                       /*swapRB*/ true,
                       /*crop*/ false);
network_.setInput(blob);
cv::Mat output = network_.forward();
```

**No letterbox.** The frame is stretched to `640×640`. Inverse scale is independent:

```
horizontal_scale = frame.cols / 640
vertical_scale   = frame.rows / 640
```

This matches a non-letterboxed export. A letterboxed ONNX would be **misaligned**.

### `candidate_rows` (anonymous namespace)

```cpp
cv::Mat candidate_rows(const cv::Mat& raw_output);
```

Normalizes the 3-D tensor to a **2-D `CV_32F` matrix of shape `[candidates × features]`**, one proposal per row.

1. Require `dims == 3`, `size[0] == 1`, `channels() == 1`.
2. `convertTo(CV_32F)`; `clone()` if not continuous.
3. Decide layout:

   ```
   features_are_first =
       size[1] >= 5 && (size[1] <= size[2] || size[2] < 5)
   ```

   So `[1, 84, 8400]` → features first (transpose to `[8400, 84]`).  
   `[1, 8400, 84]` → candidates first (clone to a 2-D view).  
   Square `[1, 84, 84]` is treated as features-first.

4. `feature_count < 5` or `candidate_count <= 0` → throw.

The transpose path wraps `output.ptr<float>()` in a `cv::Mat` header then `.t()` (which may allocate). The candidates-first path `.clone()`s. Either way the returned matrix owns / aliases float data used immediately in `detect`.

### Decode loop

For each row `[cx, cy, w, h, class…]`:

- Confidence = `class_scores[class_id]` or `*max_element(class_scores)`.
- Skip if not finite or `< confidence_threshold_`.
- Skip if box coords are non-finite or `w ≤ 0` or `h ≤ 0`.
- Map center-size in 640-space to frame:

  ```
  left   = clamp((cx - w/2) * sx, 0, frame.cols)
  top    = clamp((cy - h/2) * sy, 0, frame.rows)
  right  = clamp((cx + w/2) * sx, 0, frame.cols)
  bottom = clamp((cy + h/2) * sy, 0, frame.rows)
  ```

  then `floor` left/top, `ceil` right/bottom, skip if `right <= left` or `bottom <= top`.

Boxes and scores are pushed in lockstep, then:

```cpp
cv::dnn::NMSBoxes(boxes, confidences,
                  confidence_threshold_, nms_threshold_,
                  kept_indices);
```

Kept entries become `{boxes[i], confidences[i]}` [`DetectedObject`](detection.md)s.

**Complexity.** Forward pass dominates. Decode is `O(C · K)` for `C` candidates and `K` classes (`max_element` per row unless `class_id` is set). NMS is OpenCV’s (typically sort + greedy). `8400` candidates × 80 classes is the YOLO11n/v8n ballpark.

**Memory.** One NCHW blob `1×3×640×640` float, plus the raw output tensor, plus a `[C × F]` candidate matrix, plus `boxes`/`confidences` vectors. `network_` weights stay loaded for the lifetime of the object.

**Threading.** `cv::dnn::Net` is not safe for concurrent `detect` on one instance. Separate `YoloDetector`s (separate `readNetFromONNX`) are independent.

**Not done.** Objectness × class, letterbox + pad, class name lookup, [`ClassifiedObject`](detection.md#classifiedobject), GPU backend selection, batching (`batch` must be 1).

## Typical use

Webcam / file (`apps/main.cpp`):

```cpp
YoloDetector detector(model_path);  // all classes, 0.25 / 0.45
auto detections = detector.detect(frame);
tracker.update(detections);
```

Person-only MOT (`apps/evaluate_mot.cpp`):

```cpp
YoloDetector detector(path,
                      YoloDetector::kDefaultConfidenceThreshold,
                      YoloDetector::kDefaultNmsThreshold,
                      /*class_id*/ 0);
```

## See also

- [`DetectedObject`](detection.md#detectedobject)
- [`MotionDetector`](motion_detector.md) — no-model fallback with the same output type
- [`MultiObjectTracker`](multi_object_tracker.md)
