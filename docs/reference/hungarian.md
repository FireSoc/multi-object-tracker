# `hungarian.hpp`

Minimum-cost bipartite assignment. Used by [`MultiObjectTracker`](multi_object_tracker.md) to match predicted tracks (rows) to detections (columns).

```cpp
#include "hungarian.hpp"
```

Header depends only on `<vector>`. Implementation: `src/hungarian.cpp`.

## `HungarianAlgorithm`

```cpp
class HungarianAlgorithm {
   public:
    using CostMatrix = std::vector<std::vector<double>>;
    static std::vector<int> solve(const CostMatrix& costs);
};
```

Stateless class. There are no instance methods; constructing an object is pointless. All behavior is in `solve`.

### `CostMatrix`

```cpp
using CostMatrix = std::vector<std::vector<double>>;
```

Dense rectangular matrix. `costs[row][column]` is the cost of assigning that row to that column. **Lower is better.** Negative finite costs are legal and are minimized like any other (see tests: `{{-1, -5}, {-2, -3}}` → `{1, 0}`).

Not an Eigen type and not a `cv::Mat`.

The tracker fills this with `1.0 - IoU`, so tracker costs lie in `[0, 1]`. That convention is **not** required by `solve`.

---

### `solve`

```cpp
static std::vector<int> solve(const CostMatrix& costs);
```

**Purpose.** Compute a minimum-cost assignment of rows to columns: each row gets at most one column, each column at most one row. The objective is `Σ costs[i][assignment[i]]` over matched rows.

**Returns.** `std::vector<int>` of length `costs.size()`.

- `assignment[row]` is the chosen column index in `[0, column_count)`.
- `-1` means that row is unmatched.

Unmatched rows occur only when there are **more rows than columns** (or zero columns). When `row_count ≤ column_count` and `column_count > 0`, every row is matched. Surplus columns are simply unused.

**Parameters.**

| Name | Role |
| --- | --- |
| `costs` | Rectangular matrix. Empty (`size() == 0`) is allowed. |

**Preconditions.**

- If non-empty, every row must have the same `size()` as `costs.front()` (including `0`).
- Every entry must be finite (`std::isfinite`); `±inf` and `NaN` are rejected.

**Error behavior.**

| Condition | Exception | Message |
| --- | --- | --- |
| Ragged rows | `std::invalid_argument` | `"Hungarian cost matrix must be rectangular"` |
| Non-finite entry | `std::invalid_argument` | `"Hungarian costs must be finite"` |

Empty input and zero-column input do **not** throw:

| Input | Result |
| --- | --- |
| `costs.empty()` | `{}` |
| `n` rows, `0` columns | `n` copies of `-1` |

**Postconditions.**

- `|assignment| == row_count`.
- Every non-`-1` value is a unique column index.
- The matching is a globally optimal min-cost assignment (not greedy). The greedy-fails test `{{1,2},{2,100}}` returns `{1, 0}` (cost `4`), not `{0, 1}` (cost `101`).
- If several assignments share the same total cost, which one is returned is not specified; it is whatever this dual method produces.

**Typical use (tracker).**

```cpp
HungarianAlgorithm::CostMatrix costs(tracks.size(),
                                     std::vector<double>(detections.size()));
// costs[t][d] = 1.0 - IoU(predicted_t, detection_d)
std::vector<int> assignments = HungarianAlgorithm::solve(costs);
// assignments[t] == detection index or -1
```

Callers that care about quality (the tracker) **must still gate** the returned pairs. Hungarian will happily assign a pair with IoU `0` if that is the cheapest remaining option. `MultiObjectTracker` requires `IoU >= 0.30` after `solve`.

---

## Implementation (`src/hungarian.cpp`)

Kuhn–Munkres with dual potentials (Jonker–Volgenant-style successive shortest paths), not the textbook `O(n⁴)` mask version.

### `solve` (public entry)

1. Validate rectangular + finite.
2. `column_count == 0` → all `-1`.
3. If `row_count ≤ column_count`, call `solve_wide(costs)`.
4. Else **transpose**, `solve_wide(transposed)`, invert the matching so original rows that were not chosen stay `-1`.

Transpose is a dense `O(nm)` copy.

### `solve_wide` (anonymous namespace, not in the header)

```cpp
std::vector<int> solve_wide(const HungarianAlgorithm::CostMatrix& costs);
```

**Precondition (enforced by `solve`, not re-checked):** `row_count >= 1`, `column_count >= 1`, `row_count <= column_count`, rectangular, finite.

**Algorithm.**

- 1-based indexing; index `0` is an augmenting-path sentinel (`matched_row[0]` holds the row currently being matched).
- Dual vectors `row_potential[n+1]`, `column_potential[m+1]` as `long double`.
- For each of the `n` rows, run a Dijkstra-like search over columns using reduced costs

  `costs[r][c] − row_potential[r] − column_potential[c]`

  maintaining `minimum_slack` and `predecessor`. When the search reaches a free column (`matched_row[col] == 0`), augment along `predecessor`.
- After all rows, invert `matched_row[column] → row` into an `n`-vector of column indices (`-1` only if a row somehow stayed free; with `n ≤ m` every row is matched).

**Complexity.** Each of `n` augmentations scans columns repeatedly. Worst case `O(n m²)` for an `n × m` wide matrix (`n ≤ m`). After transpose of a tall matrix, `O(m n²)`. For the tracker, `n = #tracks`, `m = #detections` per frame — tiny.

**Numeric.** Slack and potentials are `long double`; the input stays `double`. This is the only extra precision in the library. Reduced-cost comparisons are raw `<`, no epsilon.

**Memory.** Per `solve_wide` call: potentials `O(n+m)`, `matched_row` / `predecessor` `O(m)`, and per-row `minimum_slack` + `visited` `O(m)`. No heap beyond those vectors and the optional transpose.

**Threading.** No shared state. Concurrent `solve` calls on different matrices are fine. Do not mutate `costs` during a call.

**Not provided.** Maximize-mode, sparse graphs, forbidden pairs (`inf` is rejected, not treated as “do not assign”), k-best assignments, or a cost-of-assignment accessor. Compute the sum yourself from `assignment` if you need it.

## See also

- [`MultiObjectTracker::update`](multi_object_tracker.md#update) — sole in-tree caller
- Tests: `tests/hungarian_test.cpp` (square, greedy-trap, wide, tall, empty, negative, ragged, NaN/inf, IoU-style MOT swap)
