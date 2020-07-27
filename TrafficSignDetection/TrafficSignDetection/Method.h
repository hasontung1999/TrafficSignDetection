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

struct Vec2D
{
	int iRow;
	int iCol;
};

struct Circle 
{
	int iRow;
	int iCol;
	int radius;
};

struct Line
{
	int r;
	int theta;
	Vec2D fstPnt;
	Vec2D sndPnt;
};

struct Triangle
{
	vector<Line>l;
};

#define MIN(X,Y) (X<Y?X:Y)
#define MAX(X,Y) (X>Y?X:Y)

// Parameters
const float PI = CV_PI;

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
void myAddWeight(const Mat& srcImg1, const Mat& srcImg2,Mat& dstImg);
void myInRange(const Mat& srcImg,Scalar lower,Scalar upper, Mat& dstImg);
void HSVthreshold(const Mat& srcImg, Mat& dstImg, signColor color);

float sum(vector<float>X);//Tính tổng các phần tử của vector
int gaussBlur(const Mat& srcImg, Mat& dstImg);//Lọc bằng Gaussian
vector<float>createGaussKernel(int height, int width, float sigma);//Tạo ma trận kernel Gaussian
void convolution(const Mat& srcImg, Mat& dstImg, vector<float> kernel,int sizeWindow);//Tính tích chập
void calcGradient(const Mat& img, Mat& weightOfGrad, Mat& directOfGrad);//Xác định độ lớn của đạo hàm và hướng của điểm ảnh
int detectByCanny(const Mat& srcImg, Mat& dstImg, int lowThresh, int highThresh);

vector<Circle>maxVec3D(const vector<vector<vector<int>>> &src,unsigned int thresh);
vector<Circle> CircleHoughTransform(const Mat& edgeImg,unsigned int minR,unsigned int maxR,unsigned int thresh);
vector<Circle> detectCircle(const Mat& srcImg,unsigned int minRadius,unsigned int maxRadius, unsigned int thresh,Mat& dstImg);
void cvtOneToThreeChannel(const Mat& srcImg, Mat& dstImg);

vector<Line>thresholdLine(const vector<vector<int>>& vote, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh);
vector<Line>LineHoughTransform(const Mat& edgeImg, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh);
vector<Line>detectLine(const Mat& srcImg, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh, unsigned int length, Mat& dstImg);

Vec2D findIntersect(Line l1, Line l2);
vector<Triangle>detectTriangle(vector<Line> lines,unsigned int deltaR, unsigned int deltaAngle, Mat& dstImg);


