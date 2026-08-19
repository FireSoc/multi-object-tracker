# Association

All association is inside `MultiObjectTracker::update` plus `HungarianAlgorithm::solve`. One bipartite matching per frame; no cascade, no appearance term, no Mahalanobis.

## Per-frame sequence

After every track has called `filter.predict()` (see [motion.md](motion.md)):

1. If `tracks_` or `detections` is empty, skip matching: `assignments` stays all \(-1\) (or empty).
2. Else build a dense cost matrix of size \(n_{\text{trk}} \times n_{\text{det}}\).
3. `assignments = HungarianAlgorithm::solve(costs)` — one column index per row, or \(-1\).
4. Walk tracks in order. An assignment is **accepted** only if the index is in range **and** \(\mathrm{IoU} \ge kMinimumIou\) (\(0.30\)).
5. Accepted: Kalman `update`, `missed_frames = 0`, mark that detection matched.
6. Rejected / missing: `++missed_frames` (track still holds the predicted box).
7. Unmarked detections birth tracks ([track-lifecycle.md](track-lifecycle.md)).

Hungarian therefore always proposes \(\min(n_{\text{trk}}, n_{\text{det}})\) pairs. The IoU gate is the only mechanism that can refuse a proposed pair. There are no dummy columns and no \(+\infty\) costs.

## IoU — `MultiObjectTracker::intersection_over_union`

Axis-aligned `cv::Rect2f`. Degenerate (any width/height \(\le 0\)) \(\to 0\).

\[
\mathrm{IoU}(A,B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
\]

Intersection is OpenCV `first & second` (axis-aligned overlap). If union area is 0, return 0. Predicted Kalman boxes are `cv::Rect2f`; detection boxes are `cv::Rect` promoted to `Rect2f` at the call site. YOLO integer expansion (floor/ceil) therefore slightly inflates detection area vs the network’s float box.

IoU is **not** class-aware and **not** computed in a Mahalanobis metric. Coasting tracks with \(w\) or \(h\) clamped to 0 in `to_box` get IoU 0 against everything and will fail the gate.

## Cost matrix

\[
C_{ij} = 1 - \mathrm{IoU}\bigl(\texttt{tracks\_}[i].\texttt{predicted\_box},\; \mathrm{Rect2f}(\texttt{detections}[j].\texttt{bounding\_box})\bigr)
\]

Range \([0,1]\). Disjoint boxes cost \(1\), not \(\infty\). Global assignment will still pair them if cardinality forces it; the gate then rejects.

Why \(1-\mathrm{IoU}\) rather than \(-\mathrm{IoU}\) or GIoU: Hungarian is a **minimum-cost** solver (`hungarian.cpp` uses reduced costs \(C_{ij} - u_i - v_j\)). \(1-\mathrm{IoU}\) is non-negative, which is irrelevant to correctness (the solver accepts negatives; see `SupportsNegativeFiniteCosts`) but matches the SORT convention and the unit test `AssociatesReorderedMotDetections`.

Complexity of the matrix: \(O(n_{\text{trk}} n_{\text{det}})\) IoU evaluations, each \(O(1)\).

## Hungarian / Munkres — `HungarianAlgorithm`

API (`include/hungarian.hpp`): `solve(CostMatrix) -> vector<int>` with `assignment[row] = column` or \(-1\) if that row is unmatched because there are more rows than columns.

This is Kuhn–Munkres implemented as **successive shortest paths on the equality subgraph with dual potentials**, not the 1957 starring/covering tableau. Same optimum: a perfect matching of the smaller part that minimizes \(\sum C_{i,\pi(i)}\).

### Rectangular handling — `HungarianAlgorithm::solve`

- Empty \(\to\) `{}`.
- Ragged rows or non-finite entries \(\to\) `invalid_argument`.
- Zero columns \(\to\) all \(-1\).
- \(n_{\text{row}} \le n_{\text{col}}\): `solve_wide(costs)`.
- Tall matrix: transpose, `solve_wide`, invert the matching. Unmatched original rows stay \(-1\).

`solve_wide` assumes at least as many columns as rows and therefore matches **every row**. That is why extra detections (wide) are unmatched columns, and extra tracks (tall) are unmatched rows. The tracker later reinterprets “matched but IoU \(< 0.30\)” as unmatched on **both** sides.

### `solve_wide` — potentials and slack

1-based indexing; column \(0\) is an augmenting-path sentinel (`matched_row[0] = current free row`).

Duals: `row_potential[i]` \(u_i\), `column_potential[j]` \(v_j\). Reduced cost

\[
\hat{c}_{ij} = C_{ij} - u_i - v_j.
\]

For each row \(r = 1\ldots n\), a Dijkstra-like search grows a tree of visited columns. For each unvisited column \(j\),

```
reduced_cost = costs[current_row-1][column-1] - row_potential[current_row] - column_potential[column]
minimum_slack[j] = min(minimum_slack[j], reduced_cost)
predecessor[j] = current_column   // parent in the alternating tree
```

Let \(\delta = \min_j \texttt{minimum\_slack}[j]\) over unvisited \(j\). Dual update (visited vs not):

\[
\begin{aligned}
u &\leftarrow u + \delta && \text{on rows matched to visited columns},\\
v &\leftarrow v - \delta && \text{on visited columns},\\
\text{slack} &\leftarrow \text{slack} - \delta && \text{on unvisited columns}.
\end{aligned}
\]

Stop when the chosen column is free (`matched_row[current_column] == 0`). Flip the alternating path via `predecessor`. After all rows, invert `matched_row` into `assignment`.

Arithmetic is `long double`; input costs are `double`. No \(\varepsilon\)-perturbation for ties: equal costs yield some optimal matching, not a specified one.

### Complexity

`solve_wide`: \(n\) augmentations, each visiting \(\le m+1\) columns, each visit scanning \(m\) columns \(\Rightarrow O(n m^2)\) with \(n\le m\). Tall case after transpose: \(O(n_{\text{det}}\, n_{\text{trk}}^2)\). Typical MOT: tens of tracks, so this is noise next to YOLO.

Space: \(O(n+m)\) potentials / matching plus \(O(m)\) slack per augmentation.

Correctness tests in `tests/hungarian_test.cpp`: known \(3\times 3\) optimum, greedy-trap \(2\times 2\), wide, tall (one \(-1\)), negatives, empty/ragged/NaN/\(\infty\).

## Gating (post-assignment, not in the solver)

```cpp
if (has_assignment) {
    iou = intersection_over_union(track.predicted_box, detection_box);
    if (iou >= kMinimumIou) { /* update Kalman; mark det matched */ continue; }
}
++track.missed_frames;
```

Threshold \(0.30\) (`kMinimumIou`). Applied to the **predicted** box vs the detection, not the updated box. A Hungarian pair with IoU \(0.05\) is dropped; that detection is free to birth a new ID in the same frame (`RejectsAssignmentsBelowMinimumIou`).

This is **not** Mahalanobis gating (SORT’s original optional gate on \(S = HPH^\top + R\)). Because \(C_{ij}=1\) is cheap enough to assign, crowded scenes with all IoUs below \(0.30\) still consume the Hungarian matching, then **all** those pairs fail the gate: every track coasts and every detection spawns a new ID. Putting \(\infty\) on edges with IoU \(< 0.30\) before solving would not change that unless dummy nodes exist; the code has neither.

## Unmatched tracks and detections

| Case | Track | Detection |
|---|---|---|
| Empty detections | all unmatched, miss++ | — |
| Empty tracks | — | all birth |
| Wide (\(n_{\text{det}} > n_{\text{trk}}\)) | all assigned by Hungarian | leftover columns unmatched \(\to\) birth if those columns were never marked; assigned columns may still birth if gated off |
| Tall | leftover rows `assignment[i]=-1` \(\to\) miss++ | all columns assigned; gated-off dets still birth |
| Gate fail | miss++ | `matched_detections[j]` stays false \(\to\) birth |

`matched_detections` is only set `true` on **accepted** gates. Hungarian’s leftover columns never get that flag.

Track walk is sequential in `tracks_` order; detection marks are a boolean vector. No second association pass (ByteTrack low-score, DeepSORT cascade: not here).

## Design choices (evident from code + tests)

- **Global vs greedy.** `FindsGlobalOptimumWhenGreedyFails` exists because greedy \(1-\mathrm{IoU}\) swaps IDs on the classic \(\begin{bmatrix}1&2\\2&100\end{bmatrix}\) trap. Reorder invariance is tested in `KeepsIdsWhenDetectionOrderChanges`.
- **Gate after, not before.** Keeps the solver simple (dense finite matrix) at the cost of “forced” bad pairs in low-overlap frames.
- **Predicted box, not previous posterior.** Association uses Kalman `predict()` output, so a moving object is matched where it is expected this frame, not where it was last seen.
- **No class / score in \(C\).** `area_confidence` is ignored. Two overlapping classes compete as if identical.

## Failure modes

- **ID switch at crossings** when two predicted boxes overlap similarly with both detections; IoU is permutation-ambiguous near 0.5.
- **Fragmentation** when IoU dips under 0.30 for one frame (fast motion, bad predict, detector jitter): old track coasts, new ID born, then both coexist until the old one dies at miss \(>5\).
- **Detector duplicates** that survive NMS still get two tracks (Hungarian will assign one leftover to a new birth if extra).
- **Zero-area Kalman boxes** (negative \(w,h\) clamped) never re-associate.
- **Stretch-distorted YOLO boxes** (see [detection.md](detection.md)) systematically lower IoU vs MOT GT and vs a well-shaped Kalman prior.
- Tie-breaking is arbitrary; not a bug, but IDs can flip when two IoUs are equal.
