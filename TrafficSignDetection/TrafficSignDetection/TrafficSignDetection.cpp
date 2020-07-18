#include "Method.h"

int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;
static void on_low_H_thresh_trackbar(int, void*)
{
    low_H = min(high_H - 1, low_H);
    setTrackbarPos("Low H", window_trackbar_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void*)
{
    high_H = max(high_H, low_H + 1);
    setTrackbarPos("High H", window_trackbar_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void*)
{
    low_S = min(high_S - 1, low_S);
    setTrackbarPos("Low S", window_trackbar_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void*)
{
    high_S = max(high_S, low_S + 1);
    setTrackbarPos("High S", window_trackbar_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void*)
{
    low_V = min(high_V - 1, low_V);
    setTrackbarPos("Low V", window_trackbar_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void*)
{
    high_V = max(high_V, low_V + 1);
    setTrackbarPos("High V", window_trackbar_name, high_V);
}
int main(int argc, char* argv[])
{
    const string name = "trafficSign.jpg";
    namedWindow(window_capture_name);
    namedWindow(window_threshold_name);
    namedWindow(window_trackbar_name);
    // Trackbars to set thresholds for HSV values
    createTrackbar("Low H", window_trackbar_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_trackbar_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_trackbar_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_trackbar_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_trackbar_name, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_trackbar_name, &high_V, max_value, on_high_V_thresh_trackbar);
    Mat frame, frame_HSV, frame_threshold;
    frame = imread(name, IMREAD_COLOR);
    // Convert from BGR to HSV colorspace
    cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
    // Detect the object based on HSV Range Values
    Mat image;
    image = imread("trafficSign.jpg", IMREAD_COLOR);
    imshow("Src img", image);
    cvtColor(image, image, COLOR_BGR2HSV);
    Mat redDetect;
    HSVthreshold(image, redDetect, red);
    cout << redDetect.size()<<redDetect.channels();
    imshow("Red thresh", redDetect);

    Mat oneChannel;
    cvtInrange2Gray(redDetect, oneChannel);

    Mat edge;
    detectByCanny(oneChannel, edge, 10, 50);
    imshow("Edge detect", edge);
    waitKey(0);
    return 0;
}