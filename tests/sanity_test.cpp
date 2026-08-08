// Placeholder test proving the GoogleTest wiring works.
// Replace with real tests (kalman_filter_test.cpp, hungarian_test.cpp, ...) as you build.

#include <gtest/gtest.h>
#include <Eigen/Dense>

TEST(Sanity, EigenLinks) {
    Eigen::Vector2d v(3.0, 4.0);
    EXPECT_DOUBLE_EQ(v.norm(), 5.0);
}
