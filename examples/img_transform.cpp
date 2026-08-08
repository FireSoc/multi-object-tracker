// Transforming imgs

#include <opencv2/opencv.hpp>
#include <iostream>

int grey_scale_img() {

    std::string path = "data/me.jpeg";
    cv::Mat img = cv::imread(path);
    cv::Mat img_gray, img_blur;

    // In c++ most of the time the destination img is inside function
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(img, img_blur, cv::Size(7,7), 5, 0);

    
    cv::imshow("img", img);
    cv::imshow("img gray", img_gray);
    cv::waitKey(0);


    return 0;
}

int main() {
    grey_scale_img();
    return 0;
}