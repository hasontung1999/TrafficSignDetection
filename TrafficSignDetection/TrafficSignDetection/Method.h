#pragma once
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;
using namespace std;

enum signColor
{
	red, 
	blue,
	yellow,
	black
};

// Tạo enum các hướng
enum directOfGrad
{
	East,
	East_North,
	North,
	West_North
};

//Tạo enum kiểm tra điểm biên cạnh
enum testEdgePoint
{
	Edge_Point,
	Not_Edge_Point,
	Undone
};

// Parameters
const float PI = 3.14;
const int max_value_H = 180;
const int max_value = 255;
const string window_trackbar_name = "Track Bar";
const String window_capture_name = "Image Input";
const String window_threshold_name = "Thresholding";

const Scalar minLowRed = Scalar(0, 100, 100);
const Scalar maxLowRed = Scalar(10, 255, 255);
const Scalar minUpperRed = Scalar(160, 100, 100);
const Scalar maxUpperRed = Scalar(179, 255, 255);

const Scalar minBlue = Scalar(100, 150, 0);
const Scalar maxBlue = Scalar(140, 255, 255);

const Scalar minYellow = Scalar(20, 100, 100);
const Scalar maxYellow = Scalar(30, 255, 255);

const Scalar minBlack = Scalar(0, 0, 0);
const Scalar maxBlack = Scalar(179, 255, 50);




//Method
int detectBySobel(const Mat& srcImg, Mat& dstImg);
void HSVthreshold(const Mat& srcImg, int lowH, int highH, int lowS, int highS, int lowV, int highV, Mat& dstImg);
void HSVthreshold(const Mat&srcImg,Mat& dstImg,signColor color);
void cvtInrange2Gray(const Mat& srcImg, Mat& dstImg);

float sum(vector<float>X);//Tính tổng các phần tử của vector
int gaussBlur(const Mat& srcImg, Mat& dstImg);//Lọc bằng Gaussian
vector<float>createGaussKernel(int height, int width, float sigma);//Tạo ma trận kernel Gaussian
int convolution(const Mat& srcImg, Mat& dstImg, vector<float> kernel);//Tính tích chập
void convolution(const Mat& srcImg, Mat& dstImg, vector<float> kernel,int sizeWindow);//Tính tích chập
void calcGradient(const Mat& img, Mat& weightOfGrad, Mat& directOfGrad);//Xác định độ lớn của đạo hàm và hướng của điểm ảnh
int detectByCanny(const Mat& srcImg, Mat& dstImg, int lowThresh, int highThresh);

