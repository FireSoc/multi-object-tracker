// Entry point for the multi-object tracker.
// For now this is just a smoke test that OpenCV and Eigen link correctly.

#include <iostream>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << '\n';

    Eigen::Matrix2d m;
    m << 1, 2, 3, 4;
    std::cout << "Eigen works, det = " << m.determinant() << '\n';

    return 0;
}
