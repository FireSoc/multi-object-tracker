# Track lifecycle

Lifecycle is `MultiObjectTracker` only (`include/multi_object_tracker.hpp`, `src/multi_object_tracker.cpp`). There is no tentative/confirmed split, no `min_hits`, no score, no merge, no ID pool.

Public snapshot type:

```cpp
struct TrackedObject { int id; cv::Rect2f bounding_box; int missed_frames; };
```

Internal `Track` adds `BoundingBoxKalmanFilter filter` and `predicted_box`. `update()` returns `tracked_objects_` by const ref — a copy of **every** live `Track` after birth/death this frame, including coasting ones (`missed_frames > 0`). `evaluate_mot` writes those predicted boxes to MOT CSV with confidence `1.0`.

## Frame order (authoritative)

`MultiObjectTracker::update`:

1. **Predict** all existing tracks (`filter.predict` \(\to\) `predicted_box`).
2. **Associate** (Hungarian + IoU gate). See [association.md](association.md).
3. **Update or coast:** accepted pair \(\to\) `filter.update`, `missed_frames = 0`; else `++missed_frames`.
4. **Delete:** `std::erase_if(tracks_, missed_frames > kMaximumMissedFrames)`.
5. **Birth:** every detection with `matched_detections[j] == false` \(\to\) `tracks_.emplace_back(next_id_++, box)`.
6. **Publish** `{id, predicted_box, missed_frames}` for remaining tracks (old survivors first, then newborns in detection-index order).

Birth is **after** deletion, so a gated-off detection can create a new ID in the same frame the old track is still alive (if it has miss budget left) or already gone (if this miss pushed it over 5).

## Birth

Unmatched detection \(\Rightarrow\)

```cpp
Track(int track_id, const cv::Rect2f& initial_box)
    : id(track_id), filter(initial_box), predicted_box(initial_box) {}
```

`missed_frames` defaults to 0. Kalman posterior is the detection box, \(\mathbf{v}=\mathbf{0}\), \(P=I_6\) ([motion.md](motion.md)). No `predict` on the birth frame: the published box is the raw detection (as `Rect2f`), not a filtered state.

**Every** unmatched detection births a track, including:

- first frame of a sequence,
- extra detections when \(n_{\text{det}} > n_{\text{trk}}\),
- detections whose Hungarian partner failed `kMinimumIou` (0.30).

There is no confidence gate at birth (`area_confidence` unused). YOLO’s 0.25 / NMS already happened; MOG2’s `min_area` already happened.

## Confirmation

**Not implemented.** A track is published from the birth frame (`KeepsIdsWhenDetectionOrderChanges` asserts size 2 after one `update` with two dets). SORT’s `hits` / `min_hits` / `tentative` do not exist. False-positive detections become false-positive trajectories until they miss 6 frames.

## ID assignment

`next_id_` starts at **1** (`include/multi_object_tracker.hpp`). `next_id_++` on each birth. IDs are unique for the life of the `MultiObjectTracker` instance and **never reused** after deletion. Gaps appear when tracks die. No hashing of appearance or box.

Output order is `tracks_` vector order, not sorted by ID. `apps/main.cpp` labels `ID {id}` regardless of `missed_frames`.

## Coasting

A miss is: no Hungarian column, or column present but IoU \(< 0.30\).

Then `predicted_box` stays the **already computed** Kalman prediction; `missed_frames` increments. OpenCV’s `predict()` has copied that prior into `statePost`, so the next frame predicts from the coasted state ([motion.md](motion.md)).

Published `bounding_box` during coast is that prediction (possibly off-image, possibly zero-area). Evaluation counts these as tracker hypotheses for the whole miss window (README: “Tracker boxes predicted through misses are emitted”).

`kMaximumMissedFrames = 5` is a **budget of published miss frames**, not “delete after 5 predicts” in the sense of missing the birth frame. Test `DropsTrackAfterMaximumMissedFrames`:

- Frame 0: birth, size 1, `missed_frames = 0`.
- Frames 1–5: empty `update`, size still 1 (`missed_frames` = 1…5). `5 > 5` is false.
- Frame 6: `missed_frames` becomes 6, `6 > 5`, erased, size 0.

So the track is visible on 5 consecutive empty frames and gone on the 6th. Predicate is `>` not `>=`.

## Deletion

```cpp
std::erase_if(tracks_, [](const Track& track) {
    return track.missed_frames > kMaximumMissedFrames;
});
```

Only miss-count. No “left the image”, no “\(P\) too large”, no “detection class change”. A track that keeps matching with IoU \(\ge 0.30\) lives forever (`missed_frames` reset to 0). IDs of deleted tracks are not reused; the Kalman object is destroyed with the `Track`.

## Interaction with association (fragmentation)

Typical FP/ID-switch loop this design produces:

1. IoU(pred, det) \(< 0.30\) for one frame (crossing, stretch-YOLO jitter, unmodeled \(\Delta t\)).
2. Old track: `missed_frames = 1`, still in `tracks_`.
3. Detection: unmatched \(\to\) new `next_id_`.
4. Next frames: two tracks near one object. Hungarian minimizes sum of \(1-\mathrm{IoU}\); often the new track (box = last det, \(\mathbf{v}=0\)) wins the detection, old track keeps coasting.
5. Old track dies after 5 more misses (or 5 total misses if it never rematches). Result: ID switch + a short duplicate, matching the evaluation’s ID-switch / FP counts on MOT17-05.

`RejectsAssignmentsBelowMinimumIou` is this in one step: original ID still present with `missed_frames == 1`, plus a second track.

## What is not in the lifecycle

| Mechanism | Status |
|---|---|
| Tentative / confirmed / `min_hits` | absent; birth = confirmed |
| ID recycle | absent; monotonic `next_id_` |
| Track merge / split / occlusion flag | absent |
| `time_since_update` vs `hit_streak` | only `missed_frames` |
| Removing coasted tracks from output | absent; they are drawn and evaluated |
| Class-consistent IDs | no class on `DetectedObject` |
| Maximum track count | unbounded |

## Complexity

Per frame: \(O(n_{\text{trk}})\) predict + \(O(n_{\text{trk}} n_{\text{det}})\) costs + Hungarian \(O(\min(n,m)\max(n,m)^2)\) + \(O(n_{\text{trk}} + n_{\text{det}})\) birth/death. Linear in track count for the lifecycle itself. No cap, so a noisy MOG2 camera with many blobs grows \(n_{\text{trk}}\) until misses catch up (each FP lives \(\approx 6\) frames).

## Why these constants (as evident)

- **Immediate birth** keeps the class small (no `hits` field) and matches a demo binary that should label a box as soon as it appears. Cost: detector FPs become tracks (`evaluate_mot` MOTA is FP-heavy).
- **5-frame coast** is the classic SORT-scale occlusion bandage at 15–30 FPS (\(\sim\)0.2–0.3 s). Coupled with \(F\) that does not know real \(\Delta t\), 5 frames is “5 calls”, not 5/30 s.
- **Delete after miss \(>5\)** plus **emit during coast** is why MOT evaluation is instructed to score predicted boxes through the miss window — the implementation has no “hidden tentative” buffer.
- **IDs from 1** matches MOTChallenge (`frame,id,x,y,w,h,...` with `id >= 1`); `evaluate_mot` writes `track.id` directly.
