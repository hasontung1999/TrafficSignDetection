#include "Method.h"
void myAddWeight(const Mat& srcImg1,const Mat& srcImg2,Mat& dstImg)
{
	if (srcImg1.cols != srcImg2.cols || srcImg1.rows != srcImg2.rows || srcImg1.channels() != srcImg2.channels())
	{
		cout << "Two input images MUST have same size and type!";
		return;
	}
	dstImg = Mat(srcImg1.size(), srcImg1.type());
	/*for(int i=0;i<srcImg1.rows;i++)
		for (int j = 0; j < srcImg1.cols; j++)*/
	for(int i=0;i<srcImg1.rows*srcImg1.cols;i++)
	{
		int x = i / srcImg1.cols;
		int y = i % srcImg1.cols;
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
		int x = i / srcImg.cols;
		int y = i % srcImg.cols;
		if (srcImg.channels() == 3)
		{
			Vec3b temp = srcImg.at<Vec3b>(x, y);
			if ((temp[0] >= lower[0] && temp[0] <= upper[0])
				&& (temp[1] >= lower[1] && temp[1] <= upper[1])
				&& (temp[2] >= lower[2] && temp[2] <= upper[2]))
				dstImg.at<uchar>(x, y) = 255;
			else
				dstImg.at<uchar>(x, y) = 0;
		}
		else if (srcImg.channels() == 1)
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
	int size = kernel.size();
	int xIndex, yIndex, x, y, x_, y_, sum;
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

			if (x_ < 0 || x_ >= src.rows || y_ < 0 || y_ >= src.cols)
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
	vector<float> gaussKernel = createGaussKernel(3, 3, 0.5);
	convolution(srcImg, dstImg, gaussKernel, 3);
	return 1;
}
void calcGradient(const Mat& img, Mat& weightOfGrad, Mat& directOfGrad)
{
	vector<float> sobelX1 = { -1,0,1,-2,0,2,-1,0,1 };
	vector<float> sobelX2 = { 1,0,-1,2,0,-2,1,0,-1 };
	vector<float> sobelY1 = { -1,-2,-1,0,0,0,1,2,1 };
	vector<float> sobelY2 = { 1,2,1,0,0,0,-1,-2,-1 };

	Mat directX1(img.size(), img.type()), directX2(img.size(), img.type());
	Mat directX;
	Mat directY1(img.size(), img.type()), directY2(img.size(), img.type());
	Mat directY;
	Mat tempWeight(img.size(), img.type());

	convolution(img, directX1, sobelX1, 3);
	//Sobel(img, directX, CV_8UC1, 1, 0);
	//imshow("direction X1", directX1);
	convolution(img, directX2, sobelX2, 3);
	//imshow("direction X2", directX2);

	/*for (int i = 0; i < directX.rows; i++)
	{
		for (int j = 0; j < directX.cols; j++)
		{
			cout << int(directX.at<uchar>(i, j)) << " ";
		}
		cout << endl;
	}
	cout << endl;*/

	convolution(img, directY1, sobelY1, 3);
	//Sobel(img, directY, CV_8UC1, 0, 1);
	//imshow("Direction Y1", directY1);

	convolution(img, directY2, sobelY2, 3);
	//imshow("Direction Y2", directY2);

	myAddWeight(directX1, directX2, directX);
	//imshow("Direction X", directX);
	myAddWeight(directY1, directY2, directY);
	//imshow("Direction Y", directY);

	/*for (int i = 0; i < directY.rows; i++)
	{
		for (int j = 0; j < directY.cols; j++)
		{
			cout << int(directY.at<uchar>(i, j)) << " ";
		}
		cout << endl;
	}
	cout << endl;*/
	

	Mat grad(img.size(), img.type());
	/*for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)*/
	for(int i=0;i<img.rows*img.cols;i++)
	{
		int x = i / img.cols;
		int y = i % img.cols;
		// Xác định độ lớn của đạo hàm của điểm
		weightOfGrad.at<uchar>(x, y) = saturate_cast<uchar>(sqrt(directX.at<uchar>(x, y)
			* directX.at<uchar>(x, y) + directY.at<uchar>(x, y) * directY.at<uchar>(x, y)));
		/*weightOfGrad.at<uchar>(i, j) = saturate_cast<uchar>(abs(directX.at<uchar>(i, j))
			+ abs(directY.at<uchar>(i, j)));*/
			/*tempWeight.at<uchar>(i, j) = sqrt(directX.at<uchar>(i, j)
				* directX.at<uchar>(i, j) + directY.at<uchar>(i, j) * directY.at<uchar>(i, j));*/

				/*cout<< int(sqrt(directX.at<uchar>(i, j)
					* directX.at<uchar>(i, j) + directY.at<uchar>(i, j) * directY.at<uchar>(i, j)))<<" ";*/
					/*tempWeight.at<uchar>(i, j) = abs(directX.at<uchar>(i, j))
							+ abs(directY.at<uchar>(i, j));*/
							// Xác định hướng của điểm
		double tanTheta = directY.at<uchar>(x, y) * 1.0 / directX.at<uchar>(x, y);
		//cout << "Theta="<< tanTheta;
		//grad.at<uchar>(i,j)= directY.at<uchar>(i, j) * 1.0 / directX.at<uchar>(i, j);
		//cout << directY.at<double>(i, j) << " " << directX.at<double>(i, j) << " " << tanTheta << endl;
		/*if (directX.at<double>(i, j) == 0 || (directX.at<double>(i, j) == 0 && directY.at<double>(i, j) == 0)) {
		tanTheta = None;
		}*/
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
	/*imshow("magnitude", weightOfGrad);
	imshow("grad", grad);*/
	//scaleToUchar(tempWeight, weightOfGrad);
	//Sobel(img, weightOfGrad, CV_8UC1, 1, 1);
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
	gaussBlur(srcImg, blur); // Lọc ảnh bằng Gaussian
	//imshow("blur", blur);

	Mat mask(srcImg.size(), srcImg.type());
	Mat weightOfGrad(srcImg.size(), srcImg.type()), directOfGrad(srcImg.size(), srcImg.type());
	calcGradient(blur, weightOfGrad, directOfGrad); // Xác định độ lớn của đạo hàm và hướng tại mỗi điểm của ảnh

	//imshow("magnitude", weightOfGrad);
	//imshow("grad", directOfGrad);

	/*for (int i = 0; i < blur.rows; i++)
	{
		for (int j = 0; j < blur.cols; j++)*/
	for(int i=0;i<blur.rows*blur.cols;i++)
	{
		int x = i / blur.cols;
		int y = i % blur.cols;

		int max = 0, check1 = 0, check2 = 0; // max để xác định giá trị lớn nhất theo hướng, check1 check2 để hỗ trợ
		switch (directOfGrad.at<uchar>(x, y))
		{
		case East: // Theo hướng 0
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
		case East_North: // THeo hướng 45
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
		case North: // Theo hướng 90
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
		case West_North: // Theo hướng 135
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
		max = check1 > check2 ? check1 : check2; // Xác định giá trị lớn nhất theo hướng

		if (weightOfGrad.at<uchar>(x, y) >= max) // Kiểm tra tại điểm đó có phải lớn nhất không
		{
			if (weightOfGrad.at<uchar>(x, y) > highThresh) // Nếu lớn hơn ngưỡng trên thì là biên
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
		else // Nếu điểm đấy không phải là giá trị lớn nhất thì không là biên
		{
			mask.at<uchar>(x, y) = Not_Edge_Point;
		}
	}

	vector<int> neighborX = { -1,0,1,-1,0,1,-1,0,1 };
	vector<int> neighborY = { -1,-1,-1,0,0,0,1,1,1 };
	// Dựa vào ma trận mặt nạ mask xác định biên cho ảnh output
	/*for (int i = 0; i < mask.rows; i++)
		for (int j = 0; j < mask.cols; j++)*/
	for(int i=0;i<mask.rows*mask.cols;i++)
	{
		int x = i / mask.cols;
		int y = i % mask.cols;
		if (mask.at<uchar>(x, y) == Edge_Point) //Nếu là điểm biên thì ảnh output nhận giá trị 255
			dstImg.at<uchar>(x, y) = 255;
		else if (mask.at<uchar>(x, y) == Not_Edge_Point) // Không phải biên thì nhận giá trị 0
			dstImg.at<uchar>(x, y) = 0;
		else // Xét những điểm chưa xác định: nếu theo hướng của nó mà có điểm biên ở xung quanh thì nó cũng là biên
			//và ngược lại, nếu xung quanh không có biên thì nó cũng không phải biên
		{
			//switch (directOfGrad.at<uchar>(i, j))
			//{
			//case East: // Theo hướng 0
			//	if (j != mask.cols - 1 && j != 0)
			//	{
			//		if (mask.at<uchar>(i, j + 1) == Edge_Point || mask.at<uchar>(i, j - 1) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;
			//	}
			//	else if (j == mask.cols - 1)
			//		if (mask.at<uchar>(i, j - 1) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;

			//	else
			//		if (mask.at<uchar>(i, j + 1) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;
			//	break;
			//case East_North:// Theo hướng 45
			//	if (i != mask.rows - 1 && i != 0 && j != mask.cols - 1 && j != 0)
			//	{
			//		if (mask.at<uchar>(i + 1, j - 1) == Edge_Point || mask.at<uchar>(i - 1, j + 1) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;
			//	}
			//	else if (i == mask.rows - 1 && j != mask.cols - 1)
			//		if (mask.at<uchar>(i - 1, j + 1) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;
			//	else if (i == 0 && j != 0)
			//		if (mask.at<uchar>(i + 1, j - 1) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;
			//	break;
			//case North: //Theo hướng 90
			//	if (i != mask.rows - 1 && i != 0)
			//	{
			//		if (mask.at<uchar>(i + 1, j) == Edge_Point || mask.at<uchar>(i - 1, j) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;
			//	}
			//	else if (i == mask.rows - 1)
			//		if (mask.at<uchar>(i - 1, j) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;
			//	else
			//		if (mask.at<uchar>(i + 1, j) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;

			//	break;
			//case West_North:// Theo hướng 135
			//	if (i != mask.rows - 1 && i != 0 && j != mask.cols - 1 && j != 0)
			//	{
			//		if (mask.at<uchar>(i + 1, j + 1) == Edge_Point || mask.at<uchar>(i - 1, j - 1) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;
			//	}
			//	else if (i == mask.rows - 1 && j != 0)
			//		if (mask.at<uchar>(i - 1, j - 1) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;
			//	else if (i == 0 && j != mask.cols - 1)
			//		if (mask.at<uchar>(i + 1, j + 1) == Edge_Point)
			//			dstImg.at<uchar>(i, j) = 255;
			//		else
			//			dstImg.at<uchar>(i, j) = 0;
			//	break;

			//}
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
	int iIndex, jIndex;
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
	int maxRadius = MIN(edgeImg.rows, edgeImg.cols) / 2;
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
		vector<vector<int>>iVec;
		for (int i = 0; i < edgeImg.rows; i++)
		{
			vector<int> jVec;
			for (int j = 0; j < edgeImg.cols; j++)
			{
				jVec.push_back(0);
			}
			iVec.push_back(jVec);
		}
		vote.push_back(iVec);
	}
	for (int r = minR; r <= maxR; r++)
	{
		for (int i = 0; i < edgeImg.rows * edgeImg.cols; i++)
		{
			int x = i / edgeImg.cols;
			int y = i % edgeImg.cols;
			if (edgeImg.at<uchar>(x, y) == 255)
			{
				for (int alpha = 0; alpha < 360; alpha++)
				{
					int a = x - r * sin(alpha * PI / 180);
					int b = y + r * cos(alpha * PI / 180);
					if(a>=0&&a<edgeImg.rows&&b>=0&&b<edgeImg.cols)
						vote[r][a][b] += 1;
				}
			}
		}
	}
	result = maxVec3D(vote,thresh);
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
	//int check = 0;
	//int iIndex, jIndex;
	for (int r = 0; r < vote.size(); r++)
	{
		//int maxVal = 0;
		//Vec3D tempVec3d;
		Line temp;
		for (int grad = 0; grad < vote[r].size(); grad++)
		{
			/*if (vote[r][grad] >= maxVal)
			{
				maxVal = src[r][grad];
				iIndex = r;
				jIndex = grad;
			}*/
			if (vote[r][grad] > thresh)
			{
				temp.r = r;
				temp.theta = grad;
				//tempVec3d.iCol = jIndex;
				result.push_back(temp);
			}
		}
		
	}
	return result;
}

vector<Line>LineHoughTransform(const Mat& edgeImg, unsigned int deltaR, unsigned int deltaTheta, unsigned int thresh)
{
	int diag = sqrt(pow(edgeImg.rows, 2) + pow(edgeImg.cols, 2));
	vector<vector<int>> vote;
	vector<vector<vector<int>>> position;
	vector<Line> result;

	for (int r = 0; r < diag; r++)
	{
		vector<int> tempVote;
		vector<vector<int>> tempPos1;
		for (int grad = 0; grad <= 180; grad++)
		{
			tempVote.push_back(0);
			vector<int>tempPos2;
			/*for (int i = 0; i < edgeImg.cols * edgeImg.rows; i++)
			{
				tempPos2.push_back(-1);
			}*/
			tempPos1.push_back(tempPos2);
		}
		vote.push_back(tempVote);
		position.push_back(tempPos1);
	}

	for (int i = 0; i < edgeImg.rows * edgeImg.cols; i++)
	{
		int x = i / edgeImg.cols;
		int y = i % edgeImg.cols;
		if (edgeImg.at<uchar>(x, y) == 255)
		{
			for (int grad = 0; grad <= 180; grad ++)
			{
				int realGrad = grad - 90;
				int r= x * sin(realGrad * PI / 180) + y * cos(realGrad * PI / 180);
				if (r > 0)
				{
					vote[r][grad] += 1;
					position[r][grad].push_back(i);
				}
			}
		}
	}

	result = thresholdLine(vote, deltaR, deltaTheta, thresh);
	if (result.size() != 0)
	{
		for (int i = 0; i < result.size(); i++)
		{
			int max = 0;
			int min = edgeImg.cols;

			int r = result[i].r;
			int grad = result[i].theta;
			for (int index = 0; index < position[r][grad].size(); index++)
			{
				int pos = position[r][grad][index];
				if (pos != -1)
				{
					int x = pos / edgeImg.cols;
					int y = pos % edgeImg.cols;

					if (y <= min)
					{
						min = y;
						result[i].fstPnt.iRow = x;
						result[i].fstPnt.iCol = y;
					}
					if (y >= max)
					{
						max = y;
						result[i].sndPnt.iRow = x;
						result[i].sndPnt.iCol = y;
					}
				}
			}
			result[i].theta = grad - 90;
		}
	}
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
		/*int r = lines[i].r;
		int theta = lines[i].theta-90;
		Point pt1, pt2;
		double a = cos(theta*PI/180), b = sin(theta*PI/180);
		double x0 = a * r, y0 = b * r;
		pt1.x = cvRound(x0 + length * (-b));
		pt1.y = cvRound(y0 + length * (a));
		pt2.x = cvRound(x0 - length * (-b));
		pt2.y = cvRound(y0 - length * (a));*/
		line(dstImg, Point(lines[i].fstPnt.iCol,lines[i].fstPnt.iRow), Point(lines[i].sndPnt.iCol,lines[i].sndPnt.iRow)
			, Scalar(0, 255, 0), 1);
	}
	return lines;
}

Vec2D findIntersect(Line l1, Line l2)
{
	Vec2D result;
	float r1 = l1.r;
	float r2 = l2.r;
	float theta1 = l1.theta * PI / 180;
	float theta2 = l2.theta * PI / 180;
	result.iCol = (r1 * sin(theta2) - r2 * sin(theta1)) / (cos(theta1) * sin(theta2) - cos(theta2) * sin(theta1));
	result.iRow = (r1 * cos(theta2) - r2 * cos(theta1)) / (sin(theta1) * cos(theta2) - sin(theta2) * cos(theta1));
	return result;
}

vector<Triangle>detectTriangle(vector<Line> lines,unsigned int deltaR,unsigned int deltaAngle,Mat& dstImg)
{
	vector<Triangle> candidate;
	vector<Triangle>result;
	/*vector<int>check;
	for (int i = 0; i < lines.size(); i++)
	{
		check.push_back(0);
	}*/

	for (int i = 0; i < lines.size()-2; i++)
	{
		Triangle t;
		int count = 1;
		int angle = 0;
		t.l.push_back(lines[i]);
		for (int j = i + 1; j < lines.size(); j++)
		{
			if ((abs(lines[j].theta - t.l[0].theta) > 2 * deltaAngle))
			{
				if ((abs(lines[j].theta) > 60 || abs(t.l[0].theta) > 60))
				{
					angle = abs(abs(t.l[0].theta) - abs(lines[j].theta));
				}
				else
				{
					angle = abs(t.l[0].theta) + abs(lines[j].theta);
				}

				if (count == 1)
				{
					if (abs(angle - 60) < deltaAngle)
					{
						t.l.push_back(lines[j]);
						count++;
					}
				}
				else if (count == 2)
				{
					if (abs(angle - 60) < deltaAngle && abs(t.l[1].theta - lines[j].theta) > 2 * deltaAngle)
					{
						t.l.push_back(lines[j]);
						break;
					}
				}

				/*else if (count == 2)
				{
					if (t.l[1].theta * lines[j].theta <= 0)
					{
						angle = abs(t.l[1].theta) + abs(lines[j].theta);
					}
					else
					{
						angle = abs(t.l[1].theta - lines[j].theta);
					}

					if (abs(angle - 60) < deltaAngle && abs(t.l[0].theta - lines[j].theta) > 2 * deltaAngle)
					{
						t.l.push_back(lines[j]);
						break;
					}
				}*/
			}
		}
		candidate.push_back(t);
	}

	for (int i = 0; i < candidate.size(); i++)
	{
		if (candidate[i].l.size() == 3)
			result.push_back(candidate[i]);
	}

	if (result.size() != 0)
	{
		for (int i = 0; i < result.size(); i++)
		{
			Vec2D pnt1, pnt2, pnt3;
			pnt1 = findIntersect(result[i].l[0], result[i].l[1]);
			pnt2 = findIntersect(result[i].l[1], result[i].l[2]);
			pnt3 = findIntersect(result[i].l[2], result[i].l[0]);

			line(dstImg, Point(pnt1.iCol, pnt1.iRow), Point(pnt2.iCol, pnt2.iRow), Scalar(0, 255, 0));
			line(dstImg, Point(pnt2.iCol, pnt2.iRow), Point(pnt3.iCol, pnt3.iRow), Scalar(0, 255, 0));
			line(dstImg, Point(pnt3.iCol, pnt3.iRow), Point(pnt1.iCol, pnt1.iRow), Scalar(0, 255, 0));
		}
	}

	return result;
}

