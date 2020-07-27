#include "Method.h"
int main(int argc, char* argv[])
{
    Mat image;
    image = imread("trafficSign.jpg", IMREAD_COLOR);
    imshow("Src img", image);

    Mat HSVimage;
    cvtColor(image, HSVimage, COLOR_BGR2HSV);

    Mat redDetect;
    HSVthreshold(HSVimage, redDetect, red);

    imshow("Thresholding", redDetect);

    Mat edge;
    detectByCanny(redDetect, edge, 50, 150);
    imshow("Edge detect", edge);

    /*detectCircle(edge, 50,100,200,image);
    imshow("Traffic Sign Detection", image);*/

    detectLine(edge,1,1,100, 1000, image);
    imshow("Traffic Sign Detection", image);

    /*vector<Line>lines = LineHoughTransform(edge, 1, 1, 100);
    detectTriangle(lines,10,20,image);
    imshow("Triangle", image);*/

    waitKey(0);
    return 0;
}