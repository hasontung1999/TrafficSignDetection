#include "Method.h"
int main(int argc, char* argv[])
{
    if (argc != 8)
    {
        cout << "Chuong trinh phat hien bien bao giao thong" << endl;
        cout << "<ten chuong trinh><duong dan tap tin anh><nguong duoi ban kinh><nguong tren ban kinh><do lech gia tri R><do lech gia tri Theta><nguong loc vote><mau bien bao>" << endl;
        return -1;
    }
    Mat image;
    image = imread(argv[1], IMREAD_COLOR);
    if (!image.data)
    {
        cout << "Khong the mo anh" << std::endl;
        return -1;
    }
    imshow("Src Image", image);

    unsigned int minRadius = atoi(argv[2]);
    unsigned int maxRadius = atoi(argv[3]);
    unsigned int deltaR = atoi(argv[4]);
    unsigned int deltaTheta = atoi(argv[5]);
    unsigned int thresh = atoi(argv[6]);
    unsigned int color = atoi(argv[7]);

    Mat dstImg;
    TrafficSignDetection(image, minRadius, maxRadius, deltaR, deltaTheta, thresh, color, dstImg);
    imshow("Traffic sign detection", dstImg);

    waitKey(0);
    return 0;
}