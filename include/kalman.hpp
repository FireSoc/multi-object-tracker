#pragma once

#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

class BoundingBoxKalmanFilter { 
    private:
        int kStateDimensions = 6;
        int kMeasurementDimensions = 4;
        int kControlDimensions = 0;
        int kMatrixType = CV_32F;
        float kProcessNoise = 1e-2F;
        float kMeasurementNoise = 1e-1F;
        float kInitialStateUncertainty = 1.0F;

    public:
        explicit BoundingBoxKalmanFilter(const cv::Rect2f& initial_box) : filter_(
                kStateDimensions,
                kMeasurementDimensions,
                kControlDimensions,
                kMatrixType) {

            cv::setIdentity(filter_.transitionMatrix);
            filter_.transitionMatrix.at<float>(0, 4) = 1.0F;
            filter_.transitionMatrix.at<float>(1, 5) = 1.0F;

            filter_.measurementMatrix = cv::Mat::zeros(kMeasurementDimensions, kStateDimensions, kMatrixType);
            filter_.measurementMatrix.at<float>(0, 0) = 1.0F;
            filter_.measurementMatrix.at<float>(1, 1) = 1.0F;
            filter_.measurementMatrix.at<float>(2, 2) = 1.0F;
            filter_.measurementMatrix.at<float>(3, 3) = 1.0F;

            cv::setIdentity(filter_.processNoiseCov, cv::Scalar::all(kProcessNoise));
            cv::setIdentity(filter_.measurementNoiseCov, cv::Scalar::all(kMeasurementNoise));
            cv::setIdentity(
                filter_.errorCovPost,
                cv::Scalar::all(kInitialStateUncertainty));

            filter_.statePost = cv::Mat::zeros(kStateDimensions, 1, kMatrixType);
            filter_.statePost.at<float>(0) = initial_box.x + initial_box.width / 2.0F;
            filter_.statePost.at<float>(1) = initial_box.y + initial_box.height / 2.0F;
            filter_.statePost.at<float>(2) = initial_box.width;
            filter_.statePost.at<float>(3) = initial_box.height;
        }

        cv::Rect2f predict() {
            return to_box(filter_.predict());
        }

        cv::Rect2f update(const cv::Rect2f& box) {
            cv::Mat measurement(kMeasurementDimensions, 1, kMatrixType);
            measurement.at<float>(0) = box.x + box.width / 2.0F;
            measurement.at<float>(1) = box.y + box.height / 2.0F;
            measurement.at<float>(2) = box.width;
            measurement.at<float>(3) = box.height;
            return to_box(filter_.correct(measurement));
        }

    private:
        static cv::Rect2f to_box(const cv::Mat& state) {
            const float width = std::max(0.0F, state.at<float>(2));
            const float height = std::max(0.0F, state.at<float>(3));
            const float center_x = state.at<float>(0);
            const float center_y = state.at<float>(1);
            return {center_x - width / 2.0F, center_y - height / 2.0F, width, height};
        }

        cv::KalmanFilter filter_;
};
