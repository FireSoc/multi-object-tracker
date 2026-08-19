# Motion (Kalman)

Implemented only in `include/kalman.hpp` as `BoundingBoxKalmanFilter`, a thin constructor around `cv::KalmanFilter`. There is no `.cpp`. Eigen is a CMake dependency and is unused here (`tests/sanity_test.cpp` is the only Eigen consumer).

This is the tracker’s **motion model**. `MotionDetector` (MOG2) is a detector; it does not feed this filter except insofar as its boxes become measurements.

## State and measurement

Dimensions (`kStateDimensions = 6`, `kMeasurementDimensions = 4`, `kControlDimensions = 0`, `kMatrixType = CV_32F`):

\[
\mathbf{x} =
\begin{bmatrix} c_x & c_y & w & h & v_x & v_y \end{bmatrix}^\top,
\qquad
\mathbf{z} =
\begin{bmatrix} c_x & c_y & w & h \end{bmatrix}^\top.
\]

Center-width-height, not SORT’s \((c_x, c_y, s, r)\) with area/aspect and \(\dot{s}\). There is **no** size velocity: \(w,h\) are integrated as random walks (identity block in \(F\), process noise on those diagonals).

Initialization from `initial_box` (`cv::Rect2f`, top-left + size):

\[
c_x = x + w/2,\quad c_y = y + h/2,\quad v_x = v_y = 0.
\]

`statePost` is that vector; `errorCovPost` \(P_{0|0} = I_6\) (`kInitialStateUncertainty = 1.0`).

## Linear Gaussian model as coded

OpenCV’s `cv::KalmanFilter` is the standard discrete filter. Control size 0 \(\Rightarrow\) no \(B\mathbf{u}\).

**Transition** — `setIdentity(transitionMatrix)` then

```cpp
transitionMatrix.at<float>(0, 4) = 1.0F;  // cx += vx
transitionMatrix.at<float>(1, 5) = 1.0F;  // cy += vy
```

\[
F =
\begin{bmatrix}
1 & 0 & 0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}.
\]

\[
c_x^- = c_x + v_x,\quad
c_y^- = c_y + v_y,\quad
w^- = w,\quad
h^- = h,\quad
v_x^- = v_x,\quad
v_y^- = v_y.
\]

Time unit is **one call to `predict()`**, i.e. one `MultiObjectTracker::update` / one frame. No \(\Delta t\), no FPS. Variable frame rate (webcam `waitKey(1)` vs file) is absorbed as if \(\Delta t \equiv 1\).

**Measurement** — zeros, then ones on the position/size block:

\[
H =
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0
\end{bmatrix}.
\]

**Noise** — isotropic diagonals, not SORT-style “position small / velocity large”:

\[
Q = q I_6,\quad q = 10^{-2},\qquad
R = r I_4,\quad r = 10^{-1}.
\]

(`kProcessNoise`, `kMeasurementNoise`; `setIdentity(..., Scalar::all(...))`.) Velocity process noise equals position process noise. Measurement noise on \(w,h\) equals that on \(c_x,c_y\). \(R\) is \(10\times\) larger than \(Q\)’s diagonal, so the filter trusts the process more than a single detection — coasting will not snap as hard as a detector-dominated filter, but a bad association still pulls the state because `correct` is applied whenever the IoU gate passes.

## Predict / update API

`MultiObjectTracker::update` always predicts every live track first:

```cpp
track.predicted_box = track.filter.predict();
```

`BoundingBoxKalmanFilter::predict` is `to_box(filter_.predict())`. OpenCV `predict()`:

\[
\begin{aligned}
\mathbf{x}_{k|k-1} &= F \mathbf{x}_{k-1|k-1},\\
P_{k|k-1} &= F P_{k-1|k-1} F^\top + Q,
\end{aligned}
\]

then copies `statePre`/`errorCovPre` into `statePost`/`errorCovPost`. A miss (no `update` this frame) therefore **leaves the posterior equal to this prior**. The next frame’s `predict` chains: coasting is repeated application of \(F\), \(Q\).

On an accepted association:

```cpp
track.predicted_box = track.filter.update(detection_box);
```

`update` builds \(\mathbf{z}\) from the detection’s center and size (same conversion as init) and calls `filter_.correct(measurement)`:

\[
\begin{aligned}
S &= H P_{k|k-1} H^\top + R,\\
K &= P_{k|k-1} H^\top S^{-1},\\
\mathbf{x}_{k|k} &= \mathbf{x}_{k|k-1} + K(\mathbf{z} - H\mathbf{x}_{k|k-1}),\\
P_{k|k} &= (I - KH) P_{k|k-1}.
\end{aligned}
\]

(OpenCV’s Joseph form vs simplified covariance depends on OpenCV version; the code does not override it.)

`to_box`:

\[
w' = \max(0, x_2),\quad h' = \max(0, x_3),\quad
\text{tl} = (c_x - w'/2,\; c_y - h'/2).
\]

Returned `cv::Rect2f` may lie **outside the image**; nothing clips to the frame. Negative sizes from an unlucky posterior become zero-area boxes; IoU with anything is then 0 ([association.md](association.md)).

Newborn tracks (`Track` constructor) set `predicted_box = initial_box` and **do not** call `predict` on the birth frame. First predict is the next `update`.

## What this is not

| SORT / textbook | This repo |
|---|---|
| 7D \((c_x,c_y,s,r,\dot{c}_x,\dot{c}_y,\dot{s})\) | 6D \((c_x,c_y,w,h,v_x,v_y)\) |
| Structured \(Q\) (white-noise jerk / different \(\sigma\) on \(\dot{s}\)) | \(Q = 10^{-2} I\) |
| Optional Mahalanobis gate using \(S\) | geometric IoU gate only |
| \(\Delta t\) from timestamps | \(F_{0,4}=F_{1,5}=1\) |
| Uncorrelated \(w,h\) via aspect ratio \(r\) | direct \(w,h\); aspect can drift independently |

Constant-velocity on center is why a linearly moving box still associates after a missed detection for a few frames. Constant-\(w,h\) is why a rapidly approaching object (scale change) is explained entirely by \(Q_{ww}, Q_{hh}\) and \(R\), not by a \(\dot{w}\) state — the box size lags.

## Complexity and numerics

One predict and at most one correct per track per frame: \(O(1)\) with \(n_x=6\) (OpenCV inverts \(4\times 4\) \(S\)). Dominated by detection, not this.

`CV_32F` throughout. No symmetrizing of \(P\), no PSD projection. Long coasts: \(P\) grows as \(\sum F^k Q (F^k)^\top\); velocity uncertainty grows and the next accepted \(\mathbf{z}\) can yank \(v_x,v_y\). Combined with IoU 0.30, that yank often happens only after a new ID was already born (gate failed while \(P\) was large and the prediction was wrong).

## Failure modes

- **Unmodeled \(\Delta t\).** Dropped webcam frames or `evaluate_mot` vs 30 FPS file: same \(F\). Fast objects look like process-model mismatch \(\to\) IoU miss \(\to\) new ID.
- **No \(\dot{w},\dot{h}\).** Scale change (people walking toward camera) inflates innovation on \(w,h\); \(R=0.1\) so size is not ignored, but there is no trend to extrapolate during coast.
- **Isotropic \(Q\).** Position and velocity share \(q=10^{-2}\). That is a lot of process noise on \(c_x\) relative to a 640 px frame **and** a lot on \(v\) once \(P\) inflates. Not tuned per sequence.
- **Zero init velocity.** First two observations determine \(\mathbf{v}\) through the gain. A single-frame birth then miss coasts at \(\mathbf{v}=\mathbf{0}\) (box frozen in place except \(Q\) during subsequent predicts — actually after one predict without update, \(v\) stays 0 so the box stays put). A track that never gets a second hit does not “fly off”; it sits. After a second hit, velocity is whatever \(K\) inferred from one step.
- **Unclipped boxes.** Predicted boxes can leave the image; IoU with in-frame detections goes to 0; track coasts until deletion.
- **MOG2 blobs as \(\mathbf{z}\).** Bounding rects of merged silhouettes are a different noise process than YOLO; same \(R\) is used anyway.
