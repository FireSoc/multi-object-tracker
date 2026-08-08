// Practicing with OpenCV to get comfortable with it in C++
// Filtering, Transforming, BB, and video detection are the most important ones right now
// Especially a reproducbile video output to modify

#include <opencv2/opencv.hpp>
#include <iostream>

/////////  Importing Images  ////////////

int img_import() {
    std::string path = "data/me.jpeg";

    cv::Mat img = cv::imread(path);
    cv::imshow("img", img);
    cv::waitKey(0);

    return 0;
}

/////////  Video  //////// 
void video_import() {
    std::string vid_path = "data/test_video.mp4";
    cv::VideoCapture cap(vid_path);
    cv::Mat img;

    while (true) {
        cap.read(img);

        cv::imshow("img", img);
        cv::waitKey(1);
    }

}

/////////  Webcam  //////// 
void webcam_view() {
    cv::VideoCapture cap(0);
    cv::Mat img;

    while (true) {
        cap.read(img);

        cv::imshow("img", img);
        cv::waitKey(1);
    }

}





int main() {
    // img_import();
    // video_import();
    webcam_view();

    return 0;
}