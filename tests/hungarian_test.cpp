#include "hungarian.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

TEST(HungarianAlgorithmTest, FindsKnownSquareOptimum) {
    const HungarianAlgorithm::CostMatrix costs = {
        {4.0, 1.0, 3.0},
        {2.0, 0.0, 5.0},
        {3.0, 2.0, 2.0},
    };

    const std::vector<int> assignment = HungarianAlgorithm::solve(costs);

    EXPECT_EQ(assignment, (std::vector<int>{1, 0, 2}));
    EXPECT_DOUBLE_EQ(costs[0][assignment[0]] + costs[1][assignment[1]] + costs[2][assignment[2]],
                     5.0);
}

TEST(HungarianAlgorithmTest, FindsGlobalOptimumWhenGreedyFails) {
    const HungarianAlgorithm::CostMatrix costs = {
        {1.0, 2.0},
        {2.0, 100.0},
    };

    EXPECT_EQ(HungarianAlgorithm::solve(costs), (std::vector<int>{1, 0}));
}

TEST(HungarianAlgorithmTest, SolvesWideMatrix) {
    const HungarianAlgorithm::CostMatrix costs = {
        {1.0, 4.0, 3.0},
        {2.0, 0.0, 5.0},
    };

    EXPECT_EQ(HungarianAlgorithm::solve(costs), (std::vector<int>{0, 1}));
}

TEST(HungarianAlgorithmTest, LeavesOneRowUnmatchedInTallMatrix) {
    const HungarianAlgorithm::CostMatrix costs = {
        {1.0, 9.0},
        {2.0, 3.0},
        {9.0, 1.0},
    };

    EXPECT_EQ(HungarianAlgorithm::solve(costs), (std::vector<int>{0, -1, 1}));
}

TEST(HungarianAlgorithmTest, HandlesEmptyDimensions) {
    EXPECT_TRUE(HungarianAlgorithm::solve({}).empty());

    const HungarianAlgorithm::CostMatrix no_columns(3);
    EXPECT_EQ(HungarianAlgorithm::solve(no_columns), (std::vector<int>{-1, -1, -1}));
}

TEST(HungarianAlgorithmTest, SupportsNegativeFiniteCosts) {
    const HungarianAlgorithm::CostMatrix costs = {
        {-1.0, -5.0},
        {-2.0, -3.0},
    };

    EXPECT_EQ(HungarianAlgorithm::solve(costs), (std::vector<int>{1, 0}));
}

TEST(HungarianAlgorithmTest, RejectsRaggedMatrix) {
    const HungarianAlgorithm::CostMatrix costs = {
        {1.0, 2.0},
        {3.0},
    };

    EXPECT_THROW(HungarianAlgorithm::solve(costs), std::invalid_argument);
}

TEST(HungarianAlgorithmTest, RejectsNonFiniteCosts) {
    const double infinity = std::numeric_limits<double>::infinity();
    const double quiet_nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(HungarianAlgorithm::solve({{infinity}}), std::invalid_argument);
    EXPECT_THROW(HungarianAlgorithm::solve({{quiet_nan}}), std::invalid_argument);
}

TEST(HungarianAlgorithmTest, AssociatesReorderedMotDetections) {
    // Rows are predicted tracks, columns are detections, and each cost is 1 - IoU.
    const HungarianAlgorithm::CostMatrix costs = {
        {0.90, 0.10},
        {0.05, 0.80},
    };

    EXPECT_EQ(HungarianAlgorithm::solve(costs), (std::vector<int>{1, 0}));
}
