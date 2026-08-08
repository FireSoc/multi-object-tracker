// Transforming imgs

#include <opencv2/opencv.hpp>
#include <iostream>

int grey_scale_img() {

    std::string path = "data/me.jpeg";
    cv::Mat img = cv::imread(path);
    cv::Mat img_gray, img_blur, img_canny, img_dilation, img_erode;

    // In c++ most of the time the destination img is inside function
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(img, img_blur, cv::Size(7, 7), 5, 0);
    cv::Canny(img_blur, img_canny, 50, 150);

    // Use only odd numbers here for the kernel
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(img, img_dilation, kernel);

    cv::imshow("img", img);
    cv::imshow("img canny", img_canny);
    cv::imshow("img blur", img_blur);
    cv::imshow("img dialation", img_dilation);
    cv::waitKey(0);


    return 0;
}

int main() {
    grey_scale_img();
    return 0;
}