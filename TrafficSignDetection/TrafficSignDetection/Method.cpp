#include "Method.h"
int detectBySobel(const Mat& srcImg, Mat& dstImg)
{
	if (srcImg.channels() != 1)
	{
		cout << "Error!!!Input Grayscale image.";
		return 0;
	}
	Mat inputImg(srcImg);
	Mat outputImgX(dstImg);
	Mat outputImgY(dstImg);
	vector<float> sobelX = { 1,0,-1,2,0,-2,1,0,-1 };
	vector<float> sobelY = { 1,2,1,0,0,0,-1,-2,-1 };

	gaussBlur(inputImg, inputImg);// Lọc ảnh bằng Gaussian

	convolution(inputImg, outputImgX, sobelX,3);// Biên theo hướng X
	imshow("Grad-X Edge", outputImgX);

	convolution(inputImg, outputImgY, sobelY,3); // Biên theo hướng Y
	imshow("Grad-Y Edge", outputImgY);

	for (int iRow = 0; iRow < dstImg.rows; iRow++)
		for (int iCol = 0; iCol < dstImg.cols; iCol++)
		{
			dstImg.at<uchar>(iRow, iCol) = saturate_cast<uchar>(sqrt(outputImgX.at<uchar>(iRow, iCol)
				* outputImgX.at<uchar>(iRow, iCol) + outputImgY.at<uchar>(iRow, iCol) * outputImgY.at<uchar>(iRow, iCol)));
		}

	return 1;
}
void HSVthreshold(const Mat& srcImg, int lowH, int highH, int lowS, int highS, int lowV, int highV, Mat& dstImg)
{
	inRange(srcImg, Scalar(lowH, lowS, lowV), Scalar(highH, highS, highV), dstImg);
}
void HSVthreshold(const Mat& srcImg,Mat& dstImg, signColor color)
{
	Mat lowRed, upperRed;
	switch (color)
	{
	case red:
		inRange(srcImg, minLowRed, maxLowRed, lowRed);
		inRange(srcImg, minUpperRed, maxUpperRed, upperRed);
		addWeighted(lowRed, 1.0, upperRed, 1.0, 0.0, dstImg);
		break;
	case blue:
		inRange(srcImg, minBlue, maxBlue, dstImg);
		break;
	case yellow:
		inRange(srcImg, minYellow, maxYellow, dstImg);
		break;
	case black:
		inRange(srcImg, minBlack, maxBlack, dstImg);
		break;
	default:
		break;
	}
}
void cvtInrange2Gray(const Mat& srcImg, Mat& dstImg)
{
	dstImg = Mat(srcImg.size(), CV_8UC1);
	for (int i = 0; i < dstImg.rows; i++)
		for (int j = 0; j < dstImg.cols; j++)
		{
			dstImg.at<uchar>(i, j) = (srcImg.at<Vec3b>(i, j/3)[0]
									+ srcImg.at<Vec3b>(i, j/3)[1]
									+ srcImg.at<Vec3b>(i, j/3)[2]) / 3;
			//cout << dstImg.at<uchar>(i, j);
		}
}

float sum(vector<float> X)
{
	float result = 0;
	for (int i = 0; i < X.size(); i++)
		result += X[i];
	return result;
}
vector<float> createGaussKernel(int height, int width, float sigma)
{
	// Áp dụng công thức Gaussian để tạo ma trận Gaussian
	vector<float>kernel;
	float inverse_Sqrt2Pi_Sigma;
	float inverse_2_SigmaSquare;
	int	indexRow, indexCol;

	inverse_Sqrt2Pi_Sigma = 1 / (sqrt(2 * PI) * sigma);
	inverse_2_SigmaSquare = 1 / (2 * sigma * sigma);

	for (indexRow = 0; indexRow < height; indexRow++)
	{
		for (indexCol = 0; indexCol < width; indexCol++)
		{
			kernel.push_back(inverse_Sqrt2Pi_Sigma
				* exp(-((indexRow - height / 2) * (indexRow - height / 2) + (indexCol - width / 2) * (indexCol - width / 2))
					* inverse_2_SigmaSquare));
		}
	}
	return kernel;
}
int convolution(const Mat& srcImg, Mat& dstImg, vector<float>kernel)
{
	if (srcImg.channels() != 1 && srcImg.channels() != 3)
	{
		return 0;
	}
	for (int iRow = 0; iRow < srcImg.rows; iRow++)
		for (int iCol = 0; iCol < srcImg.cols; iCol++)
		{
			vector<float> temp;
			float x;
			if (iRow == 0) //Trường hợp điểm nằm trên biên trên
			{
				if (iCol == 0) //Điểm góc trái
				{
					x = srcImg.at<uchar>(iRow, iCol) * kernel[4];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol + 1) * kernel[5];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol) * kernel[7];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol + 1) * kernel[8];
					temp.push_back(x);
				}
				else if (iCol == srcImg.cols - 1) //Điểm góc phải
				{
					x = srcImg.at<uchar>(iRow, iCol) * kernel[4];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol - 1) * kernel[3];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol) * kernel[7];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol - 1) * kernel[6];
					temp.push_back(x);
				}
				else //Các điểm còn lại trên biên trên
				{
					x = srcImg.at<uchar>(iRow, iCol) * kernel[4];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol - 1) * kernel[3];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol + 1) * kernel[5];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol) * kernel[7];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol - 1) * kernel[6];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol + 1) * kernel[8];
					temp.push_back(x);
				}
			}
			else if (iRow == srcImg.rows - 1) //Trường hợp nằm trên biên dưới
			{
				if (iCol == 0) //góc trái
				{
					x = srcImg.at<uchar>(iRow, iCol) * kernel[4];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol + 1) * kernel[5];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol) * kernel[1];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol + 1) * kernel[2];
					temp.push_back(x);
				}
				else if (iCol == srcImg.cols - 1) //góc phải
				{
					x = srcImg.at<uchar>(iRow, iCol) * kernel[4];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol - 1) * kernel[3];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol) * kernel[1];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol - 1) * kernel[0];
					temp.push_back(x);
				}
				else //Các điểm còn lại
				{
					x = srcImg.at<uchar>(iRow, iCol) * kernel[4];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol - 1) * kernel[3];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol + 1) * kernel[5];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol) * kernel[1];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol - 1) * kernel[0];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol + 1) * kernel[2];
					temp.push_back(x);
				}
			}
			else
			{
				if (iCol == 0) //Các điểm nằm trên biên trái
				{
					x = srcImg.at<uchar>(iRow, iCol) * kernel[4];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol + 1) * kernel[5];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol) * kernel[1];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol + 1) * kernel[2];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol) * kernel[7];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol + 1) * kernel[8];
					temp.push_back(x);
				}
				else if (iCol == srcImg.cols - 1) //Các điểm nằm trên biên phải
				{
					x = srcImg.at<uchar>(iRow, iCol) * kernel[4];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol - 1) * kernel[3];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol) * kernel[1];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol - 1) * kernel[0];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol) * kernel[7];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol - 1) * kernel[6];
					temp.push_back(x);
				}
				else //Các điểm còn lại
				{
					x = srcImg.at<uchar>(iRow, iCol) * kernel[4];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol - 1) * kernel[3];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow, iCol + 1) * kernel[5];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol) * kernel[1];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol - 1) * kernel[0];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow - 1, iCol + 1) * kernel[2];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol) * kernel[7];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol - 1) * kernel[6];
					temp.push_back(x);
					x = srcImg.at<uchar>(iRow + 1, iCol + 1) * kernel[8];
					temp.push_back(x);
				}
			}
			dstImg.at<uchar>(iRow, iCol) = saturate_cast<uchar>(sum(temp));
		}
	return 1;

}
void convolution(const Mat& src, Mat& dst, vector<float>kernel, int sizeWindow)
{
	dst = Mat::zeros(src.size(), src.type());
	int size = kernel.size();
	for (int i = 0; i < src.cols * src.rows; i++)
	{
		int sum = 0;
		int xIndex = i / src.cols;
		int yIndex = i % src.cols;
		for (int j = 0; j < size; j++)
		{
			int x = j / sizeWindow;
			int y = j % sizeWindow;

			int x_ = xIndex + x - sizeWindow/2;
			int y_ = yIndex + y - sizeWindow/2;

			if (x_ < 0 || x_ >= src.rows || y_ < 0 || y_ >= src.cols)
				sum += 0;
				/*x_ = 0;
			else if (x_ >= src.cols)
				x_ = src.cols - 1;
			if (y_ < 0)
				y_ = 0;
			else if (y_ >= src.rows)
				y_ = src.rows - 1;*/
			else
				sum += src.at<uchar>(x_, y_) * kernel[j];
		}
		dst.at<uchar>(xIndex, yIndex) = sum;
	}
}
int gaussBlur(const Mat& srcImg, Mat& dstImg)
{
	if (!srcImg.data)
		return 0;
	vector<float> gaussKernel = createGaussKernel(3, 3, 0.5);
	convolution(srcImg, dstImg, gaussKernel);
	return 1;
}
void calcGradient(const Mat& img, Mat& weightOfGrad, Mat& directOfGrad)
{
	vector<float> sobelX = { 1,0,-1,2,0,-2,1,0,-1 };
	vector<float> sobelY = { 1,2,1,0,0,0,-1,-2,-1 };

	Mat directX(img.size(), img.type()), directY(img.size(), img.type());
	convolution(img, directX, sobelX);
	convolution(img, directY, sobelY);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			// Xác định độ lớn của đạo hàm của điểm
			weightOfGrad.at<uchar>(i, j) = saturate_cast<uchar>(sqrt(directX.at<uchar>(i, j)
				* directX.at<uchar>(i, j) + directY.at<uchar>(i, j) * directY.at<uchar>(i, j)));

			// Xác định hướng của điểm
			double tanTheta = directY.at<double>(i, j) * 1.0 / directX.at<double>(i, j);
			if (fabs(tanTheta) >= 2.4142) {
				tanTheta = North;
			}
			else if (tanTheta <= -0.4142) {
				tanTheta = West_North;
			}
			else if (tanTheta <= 0.4142) {
				tanTheta = East;
			}
			else {
				tanTheta = East_North;
			}

			directOfGrad.at<uchar>(i, j) = tanTheta;
		}
	}
}
int detectByCanny(const Mat& srcImg, Mat& dstImg, int lowThresh, int highThresh)
{
	if (srcImg.channels() != 1)
	{
		cout << "Error!!!Input Grayscale image.";
		return 0;
	}
	dstImg = Mat::zeros(srcImg.size(), srcImg.type());
	Mat blur(srcImg.size(), srcImg.type());
	gaussBlur(srcImg, blur); // Lọc ảnh bằng Gaussian

	Mat mask(srcImg.size(), srcImg.type());
	Mat weightOfGrad(srcImg.size(), srcImg.type()), directOfGrad(srcImg.size(), srcImg.type());

	calcGradient(blur, weightOfGrad, directOfGrad); // Xác định độ lớn của đạo hàm và hướng tại mỗi điểm của ảnh

	for (int i = 0; i < blur.rows; i++)
	{
		for (int j = 0; j < blur.cols; j++)
		{

			int max = 0, check1 = 0, check2 = 0; // max để xác định giá trị lớn nhất theo hướng, check1 check2 để hỗ trợ
			switch (directOfGrad.at<uchar>(i, j))
			{
			case East: // Theo hướng 0
				if (j != blur.cols - 1 && j != 0)
				{
					check1 = weightOfGrad.at<uchar>(i, j + 1);
					check2 = weightOfGrad.at<uchar>(i, j - 1);
				}
				else if (j == blur.cols - 1)
					check2 = weightOfGrad.at<uchar>(i, j - 1);
				else
					check2 = weightOfGrad.at<uchar>(i, j + 1);
				break;
			case East_North: // THeo hướng 45
				if (i != blur.rows - 1 && i != 0 && j != blur.cols - 1 && j != 0)
				{
					check1 = weightOfGrad.at<uchar>(i + 1, j - 1);
					check2 = weightOfGrad.at<uchar>(i - 1, j + 1);
				}
				else if (i == blur.rows - 1 && j != blur.cols - 1)
					check2 = weightOfGrad.at<uchar>(i - 1, j + 1);
				else if (i == 0 && j != 0)
					check2 = weightOfGrad.at<uchar>(i + 1, j - 1);
				break;
			case North: // Theo hướng 90
				if (i != blur.rows - 1 && i != 0)
				{
					check1 = weightOfGrad.at<uchar>(i + 1, j);
					check2 = weightOfGrad.at<uchar>(i - 1, j);
				}
				else if (i == blur.rows - 1)
					check2 = weightOfGrad.at<uchar>(i - 1, j);
				else
					check2 = weightOfGrad.at<uchar>(i + 1, j);

				break;
			case West_North: // Theo hướng 135
				if (i != blur.rows - 1 && i != 0 && j != blur.cols - 1 && j != 0)
				{
					check1 = weightOfGrad.at<uchar>(i + 1, j + 1);
					check2 = weightOfGrad.at<uchar>(i - 1, j - 1);
				}
				else if (i == blur.rows - 1 && j != 0)
					check2 = weightOfGrad.at<uchar>(i - 1, j - 1);
				else if (i == 0 && j != blur.cols - 1)
					check2 = weightOfGrad.at<uchar>(i + 1, j + 1);
				break;
			}
			max = check1 > check2 ? check1 : check2; // Xác định giá trị lớn nhất theo hướng

			if (weightOfGrad.at<uchar>(i, j) >= max) // Kiểm tra tại điểm đó có phải lớn nhất không
			{
				if (weightOfGrad.at<uchar>(i, j) > highThresh) // Nếu lớn hơn ngưỡng trên thì là biên
				{
					mask.at<uchar>(i, j) = Edge_Point;
				}
				else if (weightOfGrad.at<uchar>(i, j) < lowThresh) //Nhỏ hơn ngưỡng dưới thì không là biên
				{
					mask.at<uchar>(i, j) = Not_Edge_Point;
				}
				else //Ở đoạn giữa thì phải kiểm tra lại
				{
					mask.at<uchar>(i, j) = Undone;
				}
			}
			else // Nếu điểm đấy không phải là giá trị lớn nhất thì không là biên
			{
				mask.at<uchar>(i, j) = Not_Edge_Point;
			}
		}
	}

	// Dựa vào ma trận mặt nạ mask xác định biên cho ảnh output
	for (int i = 0; i < mask.rows; i++)
		for (int j = 0; j < mask.cols; j++)
		{
			if (mask.at<uchar>(i, j) == Edge_Point) //Nếu là điểm biên thì ảnh output nhận giá trị 255
				dstImg.at<uchar>(i, j) = 255;
			else if (mask.at<uchar>(i, j) == Not_Edge_Point) // Không phải biên thì nhận giá trị 0
				dstImg.at<uchar>(i, j) = 0;
			else // Xét những điểm chưa xác định: nếu theo hướng của nó mà có điểm biên ở xung quanh thì nó cũng là biên
				//và ngược lại, nếu xung quanh không có biên thì nó cũng không phải biên
			{
				switch (directOfGrad.at<uchar>(i, j))
				{
				case East: // Theo hướng 0
					if (j != mask.cols - 1 && j != 0)
					{
						if (mask.at<uchar>(i, j + 1) == Edge_Point || mask.at<uchar>(i, j - 1) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;
					}
					else if (j == mask.cols - 1)
						if (mask.at<uchar>(i, j - 1) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;

					else
						if (mask.at<uchar>(i, j + 1) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;
					break;
				case East_North:// Theo hướng 45
					if (i != mask.rows - 1 && i != 0 && j != mask.cols - 1 && j != 0)
					{
						if (mask.at<uchar>(i + 1, j - 1) == Edge_Point || mask.at<uchar>(i - 1, j + 1) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;
					}
					else if (i == mask.rows - 1 && j != mask.cols - 1)
						if (mask.at<uchar>(i - 1, j + 1) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;
					else if (i == 0 && j != 0)
						if (mask.at<uchar>(i + 1, j - 1) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;
					break;
				case North: //Theo hướng 90
					if (i != mask.rows - 1 && i != 0)
					{
						if (mask.at<uchar>(i + 1, j) == Edge_Point || mask.at<uchar>(i - 1, j) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;
					}
					else if (i == mask.rows - 1)
						if (mask.at<uchar>(i - 1, j) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;
					else
						if (mask.at<uchar>(i + 1, j) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;

					break;
				case West_North:// Theo hướng 135
					if (i != mask.rows - 1 && i != 0 && j != mask.cols - 1 && j != 0)
					{
						if (mask.at<uchar>(i + 1, j + 1) == Edge_Point || mask.at<uchar>(i - 1, j - 1) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;
					}
					else if (i == mask.rows - 1 && j != 0)
						if (mask.at<uchar>(i - 1, j - 1) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;
					else if (i == 0 && j != mask.cols - 1)
						if (mask.at<uchar>(i + 1, j + 1) == Edge_Point)
							dstImg.at<uchar>(i, j) = 255;
						else
							dstImg.at<uchar>(i, j) = 0;
					break;

				}
			}
		}
	return 1;
}