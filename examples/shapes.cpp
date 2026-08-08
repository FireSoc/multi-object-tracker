// How to draw shapes like bounding boxes (bb)

///////////  Draw Shapes and Text  /////////////
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Blank img
    Mat img(512, 512, CV_8UC3, Scalar(255, 255, 255));
    circle(img, Point(256, 256), 155, Scalar(0, 69, 255), FILLED);

    // Rectangle with two points is bottom left and bottom right of rectangle
    rectangle(img, Point(130, 226), Point(382, 286), Scalar(255, 255, 255), 3);

    putText(img, "savir's Workshop", Point(137, 262), FONT_HERSHEY_PLAIN, 0.75, Scalar(180, 69, 140), 2);

    imshow("img", img);
    waitKey(0);


    return 0;
}