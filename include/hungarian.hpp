#pragma once

#include <vector>

class HungarianAlgorithm {
   public:
    using CostMatrix = std::vector<std::vector<double>>;

    // Returns one detection-column index per track row. A value of -1 means
    // that row is unmatched because there are more rows than columns.
    static std::vector<int> solve(const CostMatrix& costs);
};
