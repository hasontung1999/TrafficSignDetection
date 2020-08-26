#include "Method.h"
void myAddWeight(const Mat& srcImg1,const Mat& srcImg2,Mat& dstImg)
{
	if (srcImg1.cols != srcImg2.cols || srcImg1.rows != srcImg2.rows || srcImg1.channels() != srcImg2.channels())
	{
		cout << "Two input images MUST have same size and type!";
		return;
	}
	dstImg = Mat(srcImg1.size(), srcImg1.type());
	for(int i=0;i<srcImg1.rows*srcImg1.cols;i++)
	{
		int x = i / srcImg1.cols; //Giá trị hàng.
		int y = i % srcImg1.cols; //Giá trị cột.
		dstImg.at<uchar>(x, y) = saturate_cast<uchar>(srcImg1.at<uchar>(x, y) + srcImg2.at<uchar>(x, y));
	}
}

void myInRange(const Mat& srcImg,Scalar lower,Scalar upper, Mat& dstImg)
{
	if (srcImg.channels() != 1 && srcImg.channels() != 3)
	{
		cout << "Input image must have 1 or 3 channels!";
		return;
	}
	dstImg = Mat(srcImg.size(), CV_8UC1);
	for(int i=0;i<srcImg.rows*srcImg.cols;i++)
	{
		int x = i / srcImg.cols; //Giá trị hàng.
		int y = i % srcImg.cols; //Giá trị cột.
		if (srcImg.channels() == 3) //Khi ảnh input có 3 kênh
		{
			Vec3b temp = srcImg.at<Vec3b>(x, y);
			if ((temp[0] >= lower[0] && temp[0] <= upper[0])
				&& (temp[1] >= lower[1] && temp[1] <= upper[1])
				&& (temp[2] >= lower[2] && temp[2] <= upper[2]))
				dstImg.at<uchar>(x, y) = 255;
			else
				dstImg.at<uchar>(x, y) = 0;
		}
		else if (srcImg.channels() == 1) //Khi ảnh input có 1 kênh
		{
			uchar temp = srcImg.at<uchar>(x, y);
			if (temp >= lower[0] && temp <= upper[0])
				dstImg.at<uchar>(x,y) = 255;
			else
				dstImg.at<uchar>(x, y) = 0;
		}
	}
}

void HSVthreshold(const Mat& srcImg, Mat& dstImg, signColor color)
{
	Mat lowRed, upperRed;
	switch (color)
	{
	case red:
		myInRange(srcImg, minLowRed, maxLowRed, lowRed);
		myInRange(srcImg, minUpperRed, maxUpperRed, upperRed);
		myAddWeight(lowRed, upperRed, dstImg);
		break;
	case blue:
		myInRange(srcImg, minBlue, maxBlue, dstImg);
		break;
	case yellow:
		myInRange(srcImg, minYellow, maxYellow, dstImg);
		break;
	case black:
		myInRange(srcImg, minBlack, maxBlack, dstImg);
		break;
	default:
		break;
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

void convolution(const Mat& src, Mat& dst, vector<float>kernel, int sizeWindow)
{
	dst = Mat::zeros(src.size(), src.type());
	vector<int> result;
	int size = kernel.size(); //Chiều dài của vector kernel.
	int xIndex, yIndex, x, y, x_, y_, sum; /*- xIndex: lưu vị trí hàng của điểm ảnh của srcImg.
											 - yIndex: lưu vị trí cột của điểm ảnh của srcImg.
											 - x: lưu vị trí hảng của kernel.
											 - y: lưu vị trí cột của kernel.
											 - x_: lưu vị trí hàng của điểm ảnh nhận giá trị tích chập của dstImg.
											 - y_: lưu vị trí cột của điểm ảnh nhận giá trị tích chập của dstImg.
											 - sum: lưu giá trị tích chập.
										   */
	for (int i = 0; i < src.cols * src.rows; i++)
	{
		sum = 0;
		xIndex = i / src.cols;
		yIndex = i % src.cols;
		for (int j = 0; j < size; j++)
		{
			x = j / sizeWindow;
			y = j % sizeWindow;

			x_ = xIndex + x - sizeWindow / 2;
			y_ = yIndex + y - sizeWindow / 2;

			if (x_ < 0 || x_ >= src.rows || y_ < 0 || y_ >= src.cols) //Nếu vị trí của điểm nhận giá trị tích nhập
																		//vượt ra ngoài kích thước thì mặc định sum giữ nguyên.
				sum += 0;
			else
				sum += src.at<uchar>(x_, y_) * kernel[j];
		}
		dst.at<uchar>(xIndex, yIndex) = saturate_cast<uchar>(sum);
	}
}

int gaussBlur(const Mat& srcImg, Mat& dstImg)
{
	if (!srcImg.data)
		return 0;
	vector<float> gaussKernel = createGaussKernel(3, 3, 0.5); //Thực hiện việc tạo kernel Gaussian (3x3, sigma=0.5).
	convolution(srcImg, dstImg, gaussKernel, 3); //Tích chập ảnh nguồn với kernel vừa tạo để thu được output.
	return 1;
}

void calcGradient(const Mat& img, Mat& weightOfGrad, Mat& directOfGrad)
{
	vector<float> sobelX1 = { -1,0,1,-2,0,2,-1,0,1 }; //Ma trận Sobel theo hướng X từ phải qua trái.
	vector<float> sobelX2 = { 1,0,-1,2,0,-2,1,0,-1 }; //Ma trận Sobel theo hướng X từ trái qua phải.
	vector<float> sobelY1 = { -1,-2,-1,0,0,0,1,2,1 }; //Ma trận Sobel theo hướng Y từ dưới lên trên.
	vector<float> sobelY2 = { 1,2,1,0,0,0,-1,-2,-1 }; //Ma trận Sobel theo hướng Y từ trên xuống dưới.

	Mat directX1(img.size(), img.type()), directX2(img.size(), img.type());
	Mat directX;
	Mat directY1(img.size(), img.type()), directY2(img.size(), img.type());
	Mat directY;
	Mat tempWeight(img.size(), img.type());

	convolution(img, directX1, sobelX1, 3); //Xác định edge theo hướng X từ phải qua trái.
	convolution(img, directX2, sobelX2, 3); //Xác định edge theo hướng X từ trái qua phải.

	convolution(img, directY1, sobelY1, 3); //Xác định edge theo hướng Y từ dưới lên trên.
	convolution(img, directY2, sobelY2, 3); //Xác định edge theo hướng Y từ trên xuống dưới.

	myAddWeight(directX1, directX2, directX); //Xác định edge theo hướng X sẽ bằng 2 phía cộng lại.

	myAddWeight(directY1, directY2, directY); //Xác định edge theo hướng Y sẽ bằng 2 phía cộng lại.
		
	for(int i=0;i<img.rows*img.cols;i++)
	{
		int x = i / img.cols; //Lưu giá trị hàng.
		int y = i % img.cols; //Lưu giá trị cột.
		weightOfGrad.at<uchar>(x, y) = saturate_cast<uchar>(sqrt(directX.at<uchar>(x, y)
			* directX.at<uchar>(x, y) + directY.at<uchar>(x, y) * directY.at<uchar>(x, y))); //Xác định độ lớn của đạo hàm của điểm
			
		double tanTheta = directY.at<uchar>(x, y) * 1.0 / directX.at<uchar>(x, y); //Xác định hướng của điểm
		
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

		directOfGrad.at<uchar>(x, y) = tanTheta;
	}
}

int detectByCanny(const Mat& srcImg, Mat& dstImg, int lowThresh, int highThresh)
{
	if (srcImg.channels() != 1)
	{
		cout << "Error!!!Input Grayscale image.";
		return 0;
	}

	dstImg = Mat(srcImg.size(), srcImg.type());

	Mat blur(srcImg.size(), srcImg.type());
	gaussBlur(srcImg, blur); //Lọc ảnh bằng Gaussian

	Mat mask(srcImg.size(), srcImg.type());
	Mat weightOfGrad(srcImg.size(), srcImg.type()), directOfGrad(srcImg.size(), srcImg.type());
	calcGradient(blur, weightOfGrad, directOfGrad); //Xác định độ lớn của đạo hàm và hướng tại mỗi điểm của ảnh

	for(int i=0;i<blur.rows*blur.cols;i++)
	{
		int x = i / blur.cols; //Lưu giá trị hàng.
		int y = i % blur.cols; //Lưu giá trị cột.

		int max = 0, check1 = 0, check2 = 0; //max để xác định giá trị lớn nhất theo hướng, check1 check2 để hỗ trợ
		switch (directOfGrad.at<uchar>(x, y))
		{
		case East: //Theo hướng 0 độ
			if (y != blur.cols - 1 && y != 0)
			{
				check1 = weightOfGrad.at<uchar>(x, y + 1);
				check2 = weightOfGrad.at<uchar>(x, y - 1);
			}
			else if (y == blur.cols - 1)
				check2 = weightOfGrad.at<uchar>(x, y - 1);
			else
				check2 = weightOfGrad.at<uchar>(x, y + 1);
			break;
		case East_North: //Theo hướng 45 độ
			if (x != blur.rows - 1 && x != 0 && y != blur.cols - 1 && y != 0)
			{
				check1 = weightOfGrad.at<uchar>(x + 1, y - 1);
				check2 = weightOfGrad.at<uchar>(x - 1, y + 1);
			}
			else if (x == blur.rows - 1 && y != blur.cols - 1)
				check2 = weightOfGrad.at<uchar>(x - 1, y + 1);
			else if (x == 0 && y != 0)
				check2 = weightOfGrad.at<uchar>(x + 1, y - 1);
			break;
		case North: //Theo hướng 90 độ
			if (x != blur.rows - 1 && x != 0)
			{
				check1 = weightOfGrad.at<uchar>(x + 1, y);
				check2 = weightOfGrad.at<uchar>(x - 1, y);
			}
			else if (x == blur.rows - 1)
				check2 = weightOfGrad.at<uchar>(x - 1, y);
			else
				check2 = weightOfGrad.at<uchar>(x + 1, y);

			break;
		case West_North: //Theo hướng 135 độ
			if (x != blur.rows - 1 && x != 0 && y != blur.cols - 1 && y != 0)
			{
				check1 = weightOfGrad.at<uchar>(x + 1, y + 1);
				check2 = weightOfGrad.at<uchar>(x - 1, y - 1);
			}
			else if (x == blur.rows - 1 && y != 0)
				check2 = weightOfGrad.at<uchar>(x - 1, y - 1);
			else if (x == 0 && y != blur.cols - 1)
				check2 = weightOfGrad.at<uchar>(x + 1, y + 1);
			break;
		}
		max = check1 > check2 ? check1 : check2; //Xác định giá trị lớn nhất theo hướng

		if (weightOfGrad.at<uchar>(x, y) >= max) //Kiểm tra tại điểm đó có phải lớn nhất không
		{
			if (weightOfGrad.at<uchar>(x, y) > highThresh) //Nếu lớn hơn ngưỡng trên thì là biên
			{
				mask.at<uchar>(x, y) = Edge_Point;
			}
			else if (weightOfGrad.at<uchar>(x, y) < lowThresh) //Nhỏ hơn ngưỡng dưới thì không là biên
			{
				mask.at<uchar>(x, y) = Not_Edge_Point;
			}
			else //Ở đoạn giữa thì phải kiểm tra lại
			{
				mask.at<uchar>(x, y) = Undone;
			}
		}
		else //Nếu điểm đấy không phải là giá trị lớn nhất thì không là biên
		{
			mask.at<uchar>(x, y) = Not_Edge_Point;
		}
	}

	vector<int> neighborX = { -1,0,1,-1,0,1,-1,0,1 };
	vector<int> neighborY = { -1,-1,-1,0,0,0,1,1,1 };
	//Dựa vào ma trận mặt nạ mask xác định biên cho ảnh output
	for(int i=0;i<mask.rows*mask.cols;i++)
	{
		int x = i / mask.cols;
		int y = i % mask.cols;
		if (mask.at<uchar>(x, y) == Edge_Point) //Nếu là điểm biên thì ảnh output nhận giá trị 255
			dstImg.at<uchar>(x, y) = 255;
		else if (mask.at<uchar>(x, y) == Not_Edge_Point) //Không phải biên thì nhận giá trị 0
			dstImg.at<uchar>(x, y) = 0;
		else // Xét những điểm chưa xác định: nếu xung quanh của nó mà có điểm biên thì nó cũng là biên
			//và ngược lại, nếu xung quanh không có biên thì nó không phải biên
			//(có 2 cách tiếp cận: xét lân cận 8 và chỉ xét theo hướng. Đang sử dụng xét lân cận 8)
		{
			/*switch (directOfGrad.at<uchar>(i, j))
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

			}*/

			int flag = 0;
			for (int k = 0; k < 9; k++)
			{
				if (mask.at<uchar>(x + neighborX[k], y + neighborY[k]) == Edge_Point)
				{
					dstImg.at<uchar>(x, y) = 255;
					flag = 1;
					break;
				}
			}
			if (flag == 0)
				dstImg.at<uchar>(x, y) = 0;
		}
	}
	return 1;
}

vector<Circle>maxVec3D(const vector<vector<vector<int>>>& src, unsigned int thresh)
{
	vector<Circle> result;
	int check = 0;
	int iIndex, jIndex; //Lưu vị trí của maxVal;
	for (int r = 0; r < src.size(); r++)
	{
		int maxVal=0;
		Circle tempVec3d;
		for (int i = 0; i < src[r].size(); i++)
		{
			for (int j = 0; j < src[r][i].size(); j++)
			{
				if (src[r][i][j] >= maxVal)
				{
					maxVal = src[r][i][j];
					iIndex = i;
					jIndex = j;
				}
			}
		}
		if (maxVal > thresh)
		{
			tempVec3d.radius = r + 1;
			tempVec3d.iRow = iIndex;
			tempVec3d.iCol = jIndex;
			result.push_back(tempVec3d);
		}
	}
	return result;
}

vector<Circle> CircleHoughTransform(const Mat& edgeImg,unsigned int minR,unsigned int maxR, unsigned int thresh)
{
	int maxRadius = MIN(edgeImg.rows, edgeImg.cols) / 2; //Giá trị bán kính lớn nhất có thể có.
	if (minR == 0)
	{
		minR = 1;
	}
	if (maxR == 0||maxR>maxRadius)
	{
		maxR = maxRadius;
	}
	vector<Circle> result;
	vector<vector<vector<int>>> vote;

	for (int r = 0; r <= maxR; r++)
	{
		vector<vector<int>>iVec; //Temp để hỗ trợ cho khởi tạo ma trận vote.
		for (int i = 0; i < edgeImg.rows; i++)
		{
			vector<int> jVec; //Temp để hỗ trợ cho khởi tạo ma trận vote.
			for (int j = 0; j < edgeImg.cols; j++)
			{
				jVec.push_back(0);
			}
			iVec.push_back(jVec);
		}
		vote.push_back(iVec); //Khởi tạo ma trận vote ban đầu bằng 0.
	}

	for (int r = minR; r <= maxR; r++)
	{
		for (int i = 0; i < edgeImg.rows * edgeImg.cols; i++)
		{
			int x = i / edgeImg.cols; //Lưu giá trị hàng.
			int y = i % edgeImg.cols; //Lưu giá trị cột.
			if (edgeImg.at<uchar>(x, y) == 255)
			{
				for (int alpha = 0; alpha < 360; alpha++)
				{
					int a = x - r * sin(alpha * PI / 180);
					int b = y + r * cos(alpha * PI / 180);
					if(a>=0&&a<edgeImg.rows&&b>=0&&b<edgeImg.cols)
						vote[r][a][b] += 1; //Tăng vote lên 1 cho mỗi giá trị bán kính r.
				}
			}
		}
	}
	result = maxVec3D(vote,thresh); //Lọc bằng ngưỡng.
	return result;
}

void cvtOneToThreeChannel(const Mat& srcImg, Mat& dstImg)
{
	if (srcImg.channels() != 1)
	{
		cout << "Failed at cvtOneToThreeChannel function! Input must has one channel!";
		return;
	}
	dstImg = Mat(srcImg.size(), CV_8UC3);
	for (int i = 0; i < srcImg.rows * srcImg.cols; i++)
	{
		int x = i / srcImg.cols;
		int y = i % srcImg.cols;
		dstImg.at<Vec3b>(x, y)[0] = srcImg.at<uchar>(x, y);
		dstImg.at<Vec3b>(x, y)[1] = srcImg.at<uchar>(x, y);
		dstImg.at<Vec3b>(x, y)[2] = srcImg.at<uchar>(x, y);
	}
}

vector<Circle> detectCircle(const Mat& srcImg,unsigned int minRadius,unsigned int maxRadius, unsigned int thresh, Mat& dstImg)
{
	vector<Circle>circleVec = CircleHoughTransform(srcImg, minRadius, maxRadius,thresh);
	if(dstImg.channels()==1)
		cvtOneToThreeChannel(srcImg, dstImg);
	if (circleVec.size() == 0)
		return circleVec;
	for (int i = 0; i < circleVec.size(); i++)
	{
		circle(dstImg, Point(circleVec[i].iCol, circleVec[i].iRow), circleVec[i].radius, Scalar(0, 255, 0),2);
	}
	return circleVec;
}

vector<Line>thresholdLine(const vector<vector<int>>& vote, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh)
{
	vector<Line> result;
	for (int r = 0; r < vote.size(); r++)
	{
		Line temp;
		for (int grad = 0; grad < vote[r].size(); grad++)
		{
			if (vote[r][grad] > thresh)
			{
				temp.r = r;
				temp.theta = grad-180; //Theta bằng biến đếm grad - 180 là vì góc thực sự là từ -180 đến 180.
				result.push_back(temp);
			}
		}	}
	return result;
}

vector<Line>LineHoughTransform(const Mat& edgeImg, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh)
{
	int diag = sqrt(pow(edgeImg.rows, 2) + pow(edgeImg.cols, 2)); //Độ lớn đường chéo của ảnh input.
	vector<vector<int>> vote; //Mảng lưu vote của các ứng viên Line.
	vector<Line> result;

	for (int r = 0; r < diag; r++)
	{
		vector<int> tempVote;
		for (int grad = 0; grad <= 360; grad++)
		{
			tempVote.push_back(0);
		}
		vote.push_back(tempVote); //Khởi tạo mảng vote ban đầu bằng 0.
	}

	for (int i = 0; i < edgeImg.rows * edgeImg.cols; i++)
	{
		int x = i / edgeImg.cols; //Lưu vị trí hàng.
		int y = i % edgeImg.cols; //Lưu vị trí cột.
		if (edgeImg.at<uchar>(x, y) == 255)
		{
			for (int grad = 0; grad <= 360; grad ++)
			{
				int realGrad = grad - 180; //Góc thực sự dùng trong tính toán chạy từ -180 đến 180.
				int r= x * sin(realGrad * PI / 180) + y * cos(realGrad * PI / 180);
				if (r > 0)
				{
					vote[r][grad] += 1;
				}
			}
		}
	}

	result = thresholdLine(vote, deltaR, deltaTheta, thresh);
	return result;
}

vector<Line>detectLine(const Mat& srcImg, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh, unsigned int length, Mat& dstImg)
{
	vector<Line>lines = LineHoughTransform(srcImg, deltaR,deltaTheta, thresh);
	if (dstImg.channels() == 1)
		cvtOneToThreeChannel(srcImg, dstImg);
	if (lines.size() == 0)
		return lines;
	for (int i = 0; i < lines.size(); i++)
	{
		int r = lines[i].r;
		int theta = lines[i].theta;
		Point pt1, pt2;
		double a = cos(theta*PI/180), b = sin(theta*PI/180);
		double x0 = a * r, y0 = b * r;
		pt1.x = cvRound(x0 + length * (-b));
		pt1.y = cvRound(y0 + length * (a));
		pt2.x = cvRound(x0 - length * (-b));
		pt2.y = cvRound(y0 - length * (a));
		line(dstImg, pt1, pt2, Scalar(0, 255, 0), 1);
	}
	return lines;
}

Vec2D findIntersect(Line l1, Line l2)
{
	Vec2D result;
	float r1 = l1.r;
	float r2 = l2.r;
	float theta1 = l1.theta * PI / 180; //chuyển sang gradient.
	float theta2 = l2.theta * PI / 180; //chuyển sang gradient.
	result.iCol = (r1 * sin(theta2) - r2 * sin(theta1)) / (cos(theta1) * sin(theta2) - cos(theta2) * sin(theta1));
	result.iRow = (r1 * cos(theta2) - r2 * cos(theta1)) / (sin(theta1) * cos(theta2) - sin(theta2) * cos(theta1)); //Dựa vào pt đường thẳng => xác định được công thức tìm giao điểm.
	return result;
}

vector<MultiLineShape>detectTriangle(const Mat& srcImg,unsigned int deltaR,unsigned int deltaAngle, unsigned int thresh,Mat& dstImg)
{
	vector<Line> lines = LineHoughTransform(srcImg, deltaR, deltaAngle, thresh);
	vector<MultiLineShape>result; //Lưu kết quả

	for (int i = 0; i < lines.size()-2; i++)
	{
		vector<Line> listAddAngle; //Lưu danh sách các cạnh tạo với lines[i] góc 60 theo góc + góc;
		vector<Line> listSubAngle; //Lưu danh sách các cạnh tạo với lines[i] góc 60 theo góc - góc;
		for (int j = i + 1; j < lines.size(); j++)
		{
			if (abs(abs(lines[i].theta + lines[j].theta) - 60) <= deltaAngle
				&& abs(lines[i].theta-lines[j].theta)>=deltaAngle) //Tránh trường hợp 2 góc bằng nhau cộng lại bằng 60;
			{
				listAddAngle.push_back(lines[j]);
			}
			else if (abs(abs(lines[i].theta - lines[j].theta) - 60) <= deltaAngle)
			{
				listSubAngle.push_back(lines[j]);
			}
		}

		if (listAddAngle.size() > 0 && listSubAngle.size() > 0)
		{
			MultiLineShape t;
			t.line.push_back(lines[i]); //Cạnh đầu tiên của tam giác;
			for (int k = 0; k < listAddAngle.size(); k++) //Duyệt mảng listAddAngle để lấy cạnh thứ 2 của tam giác
			{
				t.line.push_back(listAddAngle[k]); //Cạnh thứ 2 của tam giác;
				for (int l = 0; l < listSubAngle.size(); l++) //Duyệt mảng listSubAngle để lấy cạnh thứ 3 của tam giác
				{
					t.line.push_back(listSubAngle[l]); //Cạnh thứ 3 của tam giác;
					result.push_back(t); //Thêm tam giác vào kết quả;
					t.line.pop_back(); //Bỏ đi cạnh thứ 3 của tam giác sau khi thêm vào kết quả 
									//để cập nhật lại biến t và tiếp tục vòng lặp;
				}
				t.line.pop_back(); //Bỏ đi cạnh thứ 2 của tam giác sau khi kết thức vòng lặp lấy cạnh thứ 3 ở trên
									//để cập nhật lại biến t và tiếp tục vòng lặp lấy cạnh thứ 2;
			}
		}
	}

	if (result.size() > 0)
	{
		for (int i = 0; i < result.size(); i++)
		{
			Vec2D pnt1, pnt2, pnt3;
			pnt1 = findIntersect(result[i].line[0], result[i].line[1]);
			pnt2 = findIntersect(result[i].line[1], result[i].line[2]);
			pnt3 = findIntersect(result[i].line[2], result[i].line[0]);

			line(dstImg, Point(pnt1.iCol, pnt1.iRow), Point(pnt2.iCol, pnt2.iRow), Scalar(0, 255, 0));
			line(dstImg, Point(pnt2.iCol, pnt2.iRow), Point(pnt3.iCol, pnt3.iRow), Scalar(0, 255, 0));
			line(dstImg, Point(pnt3.iCol, pnt3.iRow), Point(pnt1.iCol, pnt1.iRow), Scalar(0, 255, 0));
		}
	}
	return result;
}

vector<MultiLineShape> detectRectangle(const Mat& srcImg, unsigned int deltaR, unsigned int deltaAngle, unsigned int thresh, Mat& dstImg)
{
	vector<Line> lines = LineHoughTransform(srcImg, deltaR, deltaAngle, thresh);
	vector<MultiLineShape> result; //Lưu kết quả;

	for (int i = 0; i < lines.size()-3; i++)
	{
		vector<Line> list2nd; //Lưu danh sách cạnh liền thứ 2;
		vector<Line> list3rd; //Lưu danh sách cạnh liền thứ 3;
		vector<Line> list4th; //Lưu danh sách cạnh liền thứ 4;
		for (int j = i + 1; j < lines.size(); j++)
		{
			int deltaTheta = abs(lines[i].theta - lines[j].theta);
			if (abs(deltaTheta - 90) <= deltaAngle) //Xác định 2 cạnh vuông góc
			{
				if (list2nd.size() == 0 || abs(lines[j].r-list2nd[0].r)<=deltaR) //Xác định tập hợp cạnh liền thứ 2 của lines[i]
					list2nd.push_back(lines[j]);
				else
					list4th.push_back(lines[j]);
			}
			else if (abs(deltaTheta) <= deltaAngle || abs(deltaTheta - 180) <= deltaAngle) //Xác định cạnh song song
			{
				list3rd.push_back(lines[j]);
			}
		}

		if (list2nd.size() > 0 && list3rd.size() > 0 && list4th.size()>0)
		{
			MultiLineShape rect;
			rect.line.push_back(lines[i]); //Cạnh đầu tiên của hình chữ nhật;
			for (int k = 0; k < list2nd.size(); k++) //Duyệt mảng list2nd để lấy cạnh thứ 2 của hcn
			{
				rect.line.push_back(list2nd[k]); //Cạnh liền thứ 2 của hcn;
				for (int l = 0; l < list3rd.size(); l++) //Duyệt mảng list3rd để lấy cạnh thứ 3 của hcn
				{
					rect.line.push_back(list3rd[l]); //Cạnh thứ 3 của hcn;
					for (int m = 0; m < list4th.size(); m++) //Duyệt mảng list4th để lấy cạnh thứ 4 của hcn
					{
						rect.line.push_back(list4th[m]); //Cạnh thứ 4 của hcn;
						result.push_back(rect); //Thêm hcn vào kết quả;
						rect.line.pop_back(); //Bỏ đi cạnh thứ 4 của hcn sau khi thêm vào kết quả 
												//để cập nhật lại biến rect và tiếp tục vòng lặp;
					}
					rect.line.pop_back(); //Bỏ đi cạnh thứ 3 của hcn sau khi kết thức vòng lặp lấy cạnh thứ 4 ở trên
											//để cập nhật lại biến rect và tiếp tục vòng lặp lấy cạnh thứ 3;
				}
				rect.line.pop_back(); //Bỏ đi cạnh thứ 2 của hcn sau khi kết thức vòng lặp lấy cạnh thứ 3 ở trên
											//để cập nhật lại biến rect và tiếp tục vòng lặp lấy cạnh thứ 2;
			}
		}
	}

	if (result.size() > 0)
	{
		for (int i = 0; i < result.size(); i++)
		{
			Vec2D pnt1, pnt2, pnt3, pnt4;
			pnt1 = findIntersect(result[i].line[0], result[i].line[1]);
			pnt2 = findIntersect(result[i].line[1], result[i].line[2]);
			pnt3 = findIntersect(result[i].line[2], result[i].line[3]);
			pnt4 = findIntersect(result[i].line[3], result[i].line[0]);

			line(dstImg, Point(pnt1.iCol, pnt1.iRow), Point(pnt2.iCol, pnt2.iRow), Scalar(0, 255, 0));
			line(dstImg, Point(pnt2.iCol, pnt2.iRow), Point(pnt3.iCol, pnt3.iRow), Scalar(0, 255, 0));
			line(dstImg, Point(pnt3.iCol, pnt3.iRow), Point(pnt4.iCol, pnt4.iRow), Scalar(0, 255, 0));
			line(dstImg, Point(pnt4.iCol, pnt4.iRow), Point(pnt1.iCol, pnt1.iRow), Scalar(0, 255, 0));
		}
	}

	return result;
}

void TrafficSignDetection(const Mat& srcImg, unsigned int minRadius, unsigned int maxRadius, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh, unsigned int color, Mat& dstImg)
{
	dstImg = srcImg.clone();

	Mat HSVimage;
	cvtColor(srcImg, HSVimage, COLOR_BGR2HSV); //Chuyển ảnh sang hệ màu HSV

	Mat colorDetect;
	HSVthreshold(HSVimage, colorDetect, signColor(color)); //Lọc màu.
	imshow("Thresholding", colorDetect);

	Mat edge;
	detectByCanny(colorDetect, edge, 50, 150); //Phát hiện biên cạnh sử dụng Canny.
	imshow("Edge detect", edge);

	detectCircle(edge, minRadius, maxRadius, thresh, dstImg);
	detectTriangle(edge, deltaR, deltaTheta, thresh, dstImg);
	detectRectangle(edge, deltaR, deltaTheta, thresh, dstImg);
}