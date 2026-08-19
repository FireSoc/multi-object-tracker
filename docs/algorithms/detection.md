# Detection

Two frontends emit `std::vector<DetectedObject>` (`bounding_box: cv::Rect`, `area_confidence: float`). The tracker never sees class IDs, raw tensors, or masks.

## YOLO / ONNX — `YoloDetector`

Constructor (`YoloDetector::YoloDetector`) loads `cv::dnn::readNetFromONNX(model_path)`. Empty path, thresholds outside \([0,1]\), negative `class_id`, or an empty net throw. No backend/target is set: OpenCV DNN default (CPU unless the OpenCV build says otherwise).

Expected export (repo README): Ultralytics YOLOv8 / YOLO11 detection ONNX, `imgsz=640`. Live `tracker --yolo` passes no class filter; `evaluate_mot` constructs

```cpp
YoloDetector(..., kCocoPersonClassId /* 0 */);
```

so only COCO `person` scores are used.

### Preprocessing — stretch, not letterbox

`YoloDetector::detect`:

```cpp
cv::dnn::blobFromImage(frame, blob, kPixelScale, cv::Size(kInputSize, kInputSize),
                       cv::Scalar(), true, false);
```

| Argument | Value | Effect |
|---|---|---|
| `scalefactor` | `kPixelScale = 1/255` | \(\mathbf{x} \mapsto \mathbf{x}/255 \in [0,1]\) |
| `size` | \(640 \times 640\) | `kInputSize` |
| `mean` | \(0\) | no ImageNet mean subtract |
| `swapRB` | `true` | OpenCV BGR \(\to\) RGB |
| `crop` | `false` | **anisotropic resize** to 640² |

There is no letterbox, no gray pad, no scale+offset invert. Aspect ratio is destroyed whenever \(W \neq H\). Boxes are recovered with **independent** axis scales:

\[
s_x = \frac{W_{\text{frame}}}{640},\qquad s_y = \frac{H_{\text{frame}}}{640}.
\]

That pair is consistent with stretch and **inconsistent** with Ultralytics letterbox (single scale \(s = \min(640/W, 640/H)\) plus pad). On MOT17-05 (\(640\times 480\)) \(s_x=1\), \(s_y=0.75\): vertical coordinates are stretched 4/3 in network space. This is the dominant geometric mismatch vs a letterboxed export.

Complexity: blob + resize is \(O(WH)\) in the frame, then a fixed 640² inference.

### Forward and layout — `candidate_rows`

`network_.setInput(blob); network_.forward()` must return a 3D tensor `[1, A, B]`, single channel. Converted to `CV_32F`, made contiguous.

Heuristic (not a named YOLO version check):

```
features_are_first  iff  A ≥ 5  and  (A ≤ B  or  B < 5)
```

Then the candidate matrix is \(N \times F\) with \(F \ge 5\), \(N \ge 1\):

- features-first (Ultralytics default `[1, 4+C, 8400]`): `Mat(F, N).t()` \(\to\) \(N \times F\)
- candidates-first: clone as \(N \times F\)

Row layout assumed:

\[
\bigl[c_x,\; c_y,\; w,\; h,\; p_0,\; p_1,\; \ldots,\; p_{C-1}\bigr]
\]

in **640-pixel** units. No objectness channel (YOLOv8+). \(C = F - 4\). Ambiguous tensors with both \(A,B \ge 5\) and \(A \le B\) are treated as features-first; a hypothetical `[1, 100, 200]` with 200 features would be decoded wrong.

### Score, decode, clip

Per row:

1. Confidence \(p = p_{\texttt{class\_id}}\) if filtered, else \(\max_c p_c\). Drop if non-finite or \(p < \tau\) (\(\tau = 0.25\) default).
2. Drop non-finite / non-positive \(w,h\).
3. Corners in frame pixels, then clamp to \([0,W]\times[0,H]\):

\[
\begin{aligned}
x_1 &= \mathrm{clamp}((c_x - w/2)\, s_x),&
y_1 &= \mathrm{clamp}((c_y - h/2)\, s_y),\\
x_2 &= \mathrm{clamp}((c_x + w/2)\, s_x),&
y_2 &= \mathrm{clamp}((c_y + h/2)\, s_y).
\end{aligned}
\]

4. Integer box: \(\lfloor x_1\rfloor, \lfloor y_1\rfloor, \lceil x_2\rceil, \lceil y_2\rceil\). Floor/ceil **expands** the box by up to 1 px per edge. Degenerate (`right <= left`) dropped.

`class_id` out of range throws (checked inside the row loop, so a bad ID fails on the first candidate). Scores are used raw; the code assumes the ONNX graph already applied sigmoid (standard Ultralytics detect export). Raw logits would make \(\tau=0.25\) meaningless.

### NMS — `cv::dnn::NMSBoxes`

```cpp
cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, kept_indices);
```

Greedy score-sorted NMS. A box \(j\) is suppressed if \(\mathrm{IoU}(i,j) > 0.45\) with a higher-scoring kept box \(i\). Score threshold is applied again (redundant with the earlier filter). Default `eta` / `top_k` (no extra decay, no cap).

When `class_id` is unset, confidence is \(\max_c p_c\) and NMS is **class-agnostic**: a person box can kill a nearby bicycle box. When `class_id` is set, every surviving candidate is that class, so NMS is de facto class-specific.

Output: `{boxes[k], confidences[k]}` as `DetectedObject`. Class index is discarded. `area_confidence` is the class score, not area (name is leftover from MOG2).

### YOLO failure modes

- **Stretch vs letterbox.** Trained with letterbox, inferred with stretch → systematic box deformation, especially off-square video.
- **Class-agnostic NMS** in the live binary (`apps/main.cpp` does not pass `class_id`).
- **No class on `DetectedObject`.** Tracker can ID-switch across categories if the live detector is unfiltered.
- **Layout heuristic** can swap \(N\) and \(F\) on weird exports.
- **No letterbox invert** means you cannot “fix” a letterboxed model by changing only `kInputSize`.
- OpenCV DNN vs Ultralytics CUDA: numeric drift, no batching, no FP16 unless the OpenCV backend provides it.

Complexity after NMS candidates: \(O(N_{\text{cand}} \cdot C)\) for argmax, then greedy NMS \(O(M^2)\) on the \(M\) boxes above \(\tau\) (8400 is typical pre-NMS \(N\)).

## MOG2 fallback — `MotionDetector`

Used when `tracker` is run without `--yolo`. Not a motion model for tracks; it is a detector.

`cv::createBackgroundSubtractorMOG2()` with OpenCV defaults (`history=500`, `varThreshold=16`, `detectShadows=true`). `MotionDetector::detect`:

1. `subtractor->apply(frame, mask)` — 0 background, 127 shadow, 255 foreground.
2. `threshold(mask, 200, 255, THRESH_BINARY)` — drop shadows.
3. Open then close with \(5\times 5\) ellipse (`MORPH_OPEN`, `MORPH_CLOSE`).
4. `findContours(..., RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`.
5. Keep contours with `contourArea >= min_area` (500 px). Box = `boundingRect`. Confidence:

\[
p = \frac{\text{contour area}}{W H}
\]

i.e. fractional frame coverage, not a calibrated detector score. No NMS; overlapping blobs become overlapping boxes.

Failure modes: camera motion paints the whole frame as foreground; shadows leak if the 200 threshold is wrong for a given OpenCV build; `min_area=500` kills distant objects; confidence \(\propto\) area makes large blobs look “better” to a reader but the tracker **ignores** `area_confidence` entirely.
