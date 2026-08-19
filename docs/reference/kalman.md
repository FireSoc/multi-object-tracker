# `kalman.hpp`

Header-only constant-velocity Kalman filter over an axis-aligned bounding box. One instance per [`MultiObjectTracker::Track`](multi_object_tracker.md#track).

```cpp
#include "kalman.hpp"
```

Depends on `<algorithm>` and OpenCV `core` + `video/tracking` (`cv::KalmanFilter`). **Does not use Eigen**, despite `Eigen3::Eigen` being linked onto `tracker_core`.

There is no `kalman.cpp`.

## `BoundingBoxKalmanFilter`

```cpp
class BoundingBoxKalmanFilter {
   public:
    explicit BoundingBoxKalmanFilter(const cv::Rect2f& initial_box);
    cv::Rect2f predict();
    cv::Rect2f update(const cv::Rect2f& box);

   private:
    static cv::Rect2f to_box(const cv::Mat& state);
    cv::KalmanFilter filter_;
    // plus private numeric knobs (see below)
};
```

Wraps `cv::KalmanFilter` with a fixed 6-D state / 4-D measurement model. Not copy/move customized; you get the compiler defaults (OpenCV’s `KalmanFilter` is copyable, which copies the matrices).

### State and measurement

| Vector | Layout | Dim |
| --- | --- | --- |
| State `x` | `[cx, cy, w, h, vx, vy]ᵀ` | 6 |
| Measurement `z` | `[cx, cy, w, h]ᵀ` | 4 |
| Control | unused | 0 |

- `(cx, cy)` — box **center** in pixels.
- `(w, h)` — width / height in pixels.
- `(vx, vy)` — per-frame center velocity. **No** `vw`, `vh`; size is a random walk.

All matrices are `CV_32F`.

### Dynamics

Transition `F` (`transitionMatrix`), identity plus:

```
F[0,4] = 1    // cx' = cx + vx
F[1,5] = 1    // cy' = cy + vy
```

So:

```
cx ← cx + vx
cy ← cy + vy
w  ← w
h  ← h
vx ← vx
vy ← vy
```

Measurement `H` (`measurementMatrix`): 4×6 zeros with `H[i,i] = 1` for `i ∈ {0,1,2,3}` — observe center and size, not velocity.

Process noise `Q` = `1e-2 · I₆`.  
Measurement noise `R` = `1e-1 · I₄`.  
Posterior covariance `P` initialized to `1.0 · I₆`.

These are **not** tunable at runtime. Changing them requires editing the private members in the header.

No `controlMatrix`; `predict()` is called as `filter_.predict()` with no control input.

---

### Constructor

```cpp
explicit BoundingBoxKalmanFilter(const cv::Rect2f& initial_box);
```

**Purpose.** Allocate the OpenCV filter, plant `F`/`H`/`Q`/`R`/`P`, and set `statePost` from `initial_box`.

**Parameters.**

| Name | Role |
| --- | --- |
| `initial_box` | First observation, `cv::Rect2f` (`x, y, width, height` — **top-left** origin, not center). |

Initial state:

```
cx = initial_box.x + initial_box.width  / 2
cy = initial_box.y + initial_box.height / 2
w  = initial_box.width
h  = initial_box.height
vx = 0
vy = 0
```

(`statePost` is zeroed, then the four position/size entries are written.)

**Preconditions.** None checked. A degenerate box (`width`/`height` ≤ 0) is accepted; later [`to_box`](#to_box) clamps size to `≥ 0`.

**Postconditions.** The filter is ready for `predict()`. The tracker does **not** call `predict`/`update` on the spawn frame; the first emitted box for a new track is this initial rectangle, not a filtered estimate.

**Complexity.** `O(1)` matrix setup of tiny fixed size.

---

### `predict`

```cpp
cv::Rect2f predict();
```

**Purpose.** Time update: `x̂⁻ = F x̂`, `P⁻ = F P Fᵀ + Q`, then convert the predicted state to a box.

Delegates to `cv::KalmanFilter::predict()` and [`to_box`](#to_box).

**Returns.** Predicted `cv::Rect2f` (top-left + size). Width/height clamped to `≥ 0`. Center can leave the frame; there is **no** image-bound clamp.

**Preconditions.** Constructed. Calling `predict` twice without `update` coasts on the constant-velocity model (exactly what the tracker does during misses).

**Postconditions.** `filter_.statePre` / `statePost` hold the predicted state (OpenCV’s `predict` copies pre → post when you do not immediately `correct`). The tracker stores the returned box as `Track::predicted_box`.

**Complexity.** OpenCV’s predict for `n=6` is `O(n³)` in principle (`F P Fᵀ`); irrelevant at this size.

---

### `update`

```cpp
cv::Rect2f update(const cv::Rect2f& box);
```

**Purpose.** Measurement update (`correct`) with a detected box.

**Parameters.**

| Name | Role |
| --- | --- |
| `box` | Detection in the same top-left `cv::Rect2f` convention. Converted to `[cx, cy, w, h]`. |

Builds a 4×1 `CV_32F` measurement and calls `filter_.correct(measurement)`, then [`to_box`](#to_box).

**Returns.** Corrected box (may differ from `box`; that is the point).

**Preconditions.** Usually one `predict()` since the last `update` (standard Kalman). OpenCV will still `correct` without it; the tracker always `predict`s all tracks before associating, then `update`s matches.

**Postconditions.** Velocity is inferred from the innovation. Size is pulled toward the measurement with `R` vs `Q` as above (`R` is 10× `Q` on the diagonal, so measurements are trusted less than the process model on a per-variance reading — in practice both are large relative to pixel boxes and the filter is loosely tuned).

**Does not** check finiteness or positive size of `box`.

---

### `to_box` (private static)

```cpp
static cv::Rect2f to_box(const cv::Mat& state);
```

```
w = max(0, state[2])
h = max(0, state[3])
return {state[0] - w/2, state[1] - h/2, w, h}
```

Negative Kalman sizes become a zero-area box centered on `(cx, cy)`. Used by both `predict` and `update`.

**Preconditions.** `state` is a 6×1 (or at least 4×1) `float` matrix; not validated. Passing anything else is undefined (`at<float>`).

---

### Private knobs

Declared as **non-static data members** with default initializers, not `static constexpr`. They are initialized before `filter_` (declaration order) so the constructor initializer list may read them.

| Member | Value |
| --- | --- |
| `kStateDimensions` | `6` |
| `kMeasurementDimensions` | `4` |
| `kControlDimensions` | `0` |
| `kMatrixType` | `CV_32F` |
| `kProcessNoise` | `1e-2F` |
| `kMeasurementNoise` | `1e-1F` |
| `kInitialStateUncertainty` | `1.0F` |

`filter_` is the wrapped `cv::KalmanFilter`.

---

## Typical use

Only [`MultiObjectTracker`](multi_object_tracker.md) constructs these:

```cpp
// spawn
Track(id, detection_rect)  // filter(initial_box), predicted_box = initial_box

// each frame, all tracks:
track.predicted_box = track.filter.predict();

// matched and IoU ≥ 0.30:
track.predicted_box = track.filter.update(detection_box);

// miss: skip update; predicted_box stays at predict()
```

Do not share one `BoundingBoxKalmanFilter` across identities. Do not call `update` with a box from a different object; there is no gating inside this class.

## Threading / memory

Not thread-safe (mutates `filter_`). Each track owns its own instance. Per-object footprint is the OpenCV Kalman matrices (`6×6`, `4×6`, `4×4`, two `6×1` states) — tens of floats.

## See also

- [`MultiObjectTracker::Track`](multi_object_tracker.md#track)
- OpenCV `cv::KalmanFilter` (`predict` / `correct`)
