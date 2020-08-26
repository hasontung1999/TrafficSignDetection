#pragma once
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;
using namespace std;

//Tạo enum màu của biển báo
enum signColor
{
	red, 
	blue,
	yellow,
	black
};

//Tạo enum các hướng (Canny)
enum directOfGrad
{
	East,
	East_North,
	North,
	West_North
};

//Tạo enum kiểm tra điểm biên cạnh (Canny)
enum testEdgePoint
{
	Edge_Point,
	Not_Edge_Point,
	Undone
};

//Tạo stuct tọa độ điểm ảnh
struct Vec2D 
{
	int iRow;
	int iCol;
};

//Tạo struct hình tròn gồm tọa độ tâm và bán kính
struct Circle 
{
	int iRow;
	int iCol;
	int radius;
};

//Sử dụng phương trình đường thẳng dạng r=x*cos(theta) + y*sin(theta)
struct Line
{
	int r;
	int theta;
};

//Tạo struct hình gồm nhiều đường thẳng (tam giác, hcn, ...)
struct MultiLineShape
{
	vector<Line> line;
};

#define MIN(X,Y) (X<Y?X:Y)
#define MAX(X,Y) (X>Y?X:Y)

/*~~~~~~~~~~~~~~~~~PARAMETERS~~~~~~~~~~~~~~~~~~~~*/
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
const Scalar maxBlack = Scalar(179, 255, 50); //Những scalar vùng màu tương ứng theo enum màu biển báo ở trên
												//theo không gian HSV


/*~~~~~~~~~~~~~~~~~METHOD~~~~~~~~~~~~~~~~~~~~*/

/*Hàm myAddWeight (ảnh xám):
- Input: + srcImg1: ảnh thứ nhất.
		 + srcImg2: ảnh thứ hai.
		 + dstImg: ảnh lưu kết quả.
- Output: hàm Void, output lưu vào dstImg (mỗi điểm ảnh có giá trị bằng tổng của 
											giá trị của 2 ảnh input tại vị trí điểm ảnh tương ứng)
*/
void myAddWeight(const Mat& srcImg1, const Mat& srcImg2,Mat& dstImg);

/*Hàm myInRange:
- Input: + srcImg: ảnh nguồn input (1 kênh hoặc 3 kênh).
		 + lower: scalar mang giá trị ngưỡng dưới.
		 + upper: scalar mang giá trị ngưỡng trên.
		 + dstImg: ảnh lưu kết quả (1 kênh).
- Output: hàm Void, output lưu vào dstImg (kết quả là ảnh nhị phân, tại mỗi điểm ảnh mang giá trị:
											255: khi giá trị điểm ảnh trong ngưỡng.
											0: khi giá trị điểm ảnh ngoài ngưỡng.
*/
void myInRange(const Mat& srcImg,Scalar lower,Scalar upper, Mat& dstImg);

/*Hàm HSTthreshold:
- Input: + srcImg: ảnh input.
		 + dstImg: ảnh lưu kết quả.
		 + color: màu cần lấy.
- Output: hàm Void, output lưu vào dstImg (tương tự hàm myInRange).
*/
void HSVthreshold(const Mat& srcImg, Mat& dstImg, signColor color);

/*Tính tổng các phần tử của vector: (Hàm này không sử dụng trong bài)
- Input: X: vector float.
- Output: giá trị float là tổng các phần tử của vector.
*/
float sum(vector<float>X);

/*Lọc bằng Gaussian:
- Input: + srcImg: ảnh nguồn input.
		 + dstImg: ảnh output (ảnh sau khi đã được làm nhòe).
- Output: 1 hoặc 0 cho việc thực hiện được hoặc không.
*/
int gaussBlur(const Mat& srcImg, Mat& dstImg);

/*Hàm createGaussKernel:
- Input: + height: chiều cao ma trận.
		 + width: chiều dài ma trận.
		 + sigma: giá trị sigma trong công thức Gaussian.
- Output: là vector 1 chiều float chứa giá trị của ma trận Gaussian.
*/
vector<float>createGaussKernel(int height, int width, float sigma);

/*Hàm convolution:
- Input: + srcImg: ảnh nguồn input.
		 + dstImg: ảnh output.
		 + kernel: kernel tích chập.
		 + sizeWindow: kích thước của kernel (ví dụ: kernel 3x3 thì sizeWindow=3).
- Output: hàm void, output lưu vào dstImg (lưu kết quả tích chập srcImg với kernel).
*/
void convolution(const Mat& srcImg, Mat& dstImg, vector<float> kernel,int sizeWindow);

/*Hàm calcGradient:
- Input: + img: ảnh nguồn input.
		 + weightOfGrad: chứa chiều dài (độ lớn) của vector hướng.
		 + directOfGrad: chứa hướng của vector hướng (góc).
- Output: hàm void, output lưu vào weightOfGrad và directOfGrad.
*/
void calcGradient(const Mat& img, Mat& weightOfGrad, Mat& directOfGrad);

/*Hàm detectByCanny:
- Input: + srcImg: ảnh nguồn input.
		 + dstImg: ảnh output (chứa kết quả biên cạnh).
		 + lowThresh: ngưỡng dưới.
		 + highThresh: ngưỡng trên.
- Output: 1 hoặc 0 thể hiện việc thực hiên thành công hay không.
*/
int detectByCanny(const Mat& srcImg, Mat& dstImg, int lowThresh, int highThresh);

/*Hàm maxVec3D:
- Input: + src: vector 3 chiều.
		 + thresh: ngưỡng lọc.
- Output: một mảng Circle lưu kết quả có vote lớn nhất và lớn hơn ngưỡng.
*/
vector<Circle>maxVec3D(const vector<vector<vector<int>>> &src,unsigned int thresh);

/*Hàm CircleHoughTransform:
- Input: + edgeImg: ảnh biên cạnh.
		 + minR: ngưỡng dưới giá trị bán kính.
		 + maxR: ngưỡng trên giá trị bán kính.
		 + thresh: ngưỡng lọc vote.
- Output: môt mảng Circle lưu kết quả các đường tròn phát hiện được trong edgeImg.
*/
vector<Circle> CircleHoughTransform(const Mat& edgeImg,unsigned int minR,unsigned int maxR,unsigned int thresh);

/*Hàm detectCircle:
- Input: + srcImg: ảnh nguồn input.
		 + minRadius: ngưỡng dưới bán kính.
		 + maxRadius: ngưỡng trên bán kính.
		 + thresh: ngưỡng lọc vote.
		 + dstImg: ảnh output.
- Output: một mảng Circle lưu kết quả các đường tròn phát hiên được trong srcImg.
*/
vector<Circle> detectCircle(const Mat& srcImg,unsigned int minRadius,unsigned int maxRadius, unsigned int thresh,Mat& dstImg);

/*Hàm cvtOneToThreeChannel:
- Input: + srcImg: ảnh nguồn input (1 kênh).
		 + dstImg: ảnh output (3 kênh).
- Output: hàm void, output lưu vào dstImg (có 3 kênh mà giá trị mỗi kênh bằng giá trị của srcImg tại vị trí tương ứng).
*/
void cvtOneToThreeChannel(const Mat& srcImg, Mat& dstImg);

/*Hàm thresholdLine:
- Input: + vote: mảng chứa vote của những ứng viên đường thẳng.
		 + deltaR: độ sai lệch của giá trị r.
		 + deltaTheta: độ sai lệch của giá trị góc theta.
		 + thresh: giá trị ngưỡng.
- Output: mảng Line có vote lớn hơn ngưỡng thresh.
*/
vector<Line>thresholdLine(const vector<vector<int>>& vote, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh);

/*LineHoughTransform:
- Input: + edgeImg: ảnh biên cạnh input.
		 + deltaR: độ sai lệch của giá trị r.
		 + deltaTheta: độ sai lệch của giá trị góc theta.
		 + thresh: giá trị ngưỡng.
- Output: mảng Line phát hiên được trong ảnh input.
*/
vector<Line>LineHoughTransform(const Mat& edgeImg, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh);

/*Hàm detectLine:
- Input: + srcImg: ảnh nguồn input.
		 + deltaR: độ sai lệch của giá trị r.
		 + deltaTheta: độ sai lệch của giá trị góc theta.
		 + thresh: ngưỡng lọc vote.
		 + length: chiều dài của đường thẳng.
		 + dstImg: ảnh output.
- Output: mảng Line phát hiện được trong ảnh srcImg.
*/
vector<Line>detectLine(const Mat& srcImg, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh, unsigned int length, Mat& dstImg);

/*Hàm findIntersect:
- Input: + l1: Line thứ nhất.
		 + l2: Line thứ hai.
- Output: tọa độ giao điểm của 2 đường thẳng.
*/
Vec2D findIntersect(Line l1, Line l2);

/*Hàm detectTriangle:
- Input: + srcImg: ảnh nguồn input.
		 + deltaR: độ sai lệch giá trị r.
		 + deltaAngle: độ sai lệch giá trị góc theta.
		 + thresh: ngưỡng lọc.
		 + dstImg: ảnh output.
- Output: mảng các hình tam giác phát hiện được từ danh sách các line.
*/
vector<MultiLineShape>detectTriangle(const Mat& srcImg, unsigned int deltaR, unsigned int deltaAngle, unsigned int thresh, Mat& dstImg);

/*Hàm detectRectangle:
- Input: + srcImg: ảnh nguồn input.
		 + deltaR: độ sai lệch giá trị r.
		 + deltaAngle: độ sai lệch giá trị góc theta.
		 + thresh: ngưỡng lọc.
		 + dstImg: ảnh output.
- Output: mảng các hình chữ nhật phát hiện được từ danh sách các line.
*/
vector<MultiLineShape> detectRectangle(const Mat& srcImg, unsigned int deltaR, unsigned int deltaAngle, unsigned int thresh, Mat& dstImg);

/*Hàm TrafficSignDetection:
- Input: + srcImg: ảnh nguồn input (3 kênh màu RGB).
		 + minRadius: ngưỡng dưới của bán kính phát hiên hình tròn.
		 + maxRaidus: ngưỡng trên của bán kính phát hiện hình tròn.
		 + deltaR: độ lệch giá trị r trong việc phát hiện đường thẳng.
		 + deltaTheta: độ lệch giá trị góc trong việc phát hiện đường thẳng.
		 + thresh: ngưỡng lọc vote của các ứng viên phát hiện.
		 + color: màu của việc phân ngưỡng để phát hiện biển báo.
		 + dstImg: ảnh output chứa hình ảnh phát hiện biển báo.
- Output: hàm void, output chứa trong dstImg.*/
void TrafficSignDetection(const Mat& srcImg, unsigned int minRadius, unsigned int maxRadius, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh, unsigned int color, Mat& dstImg);


