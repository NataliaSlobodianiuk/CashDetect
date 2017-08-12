#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	Mat src = imread(argv[1]);

	if (!src.data)
		return -1;

	imshow("Source Image", src);

	for (int x = 0; x < src.rows; x++) 
	{
		for (int y = 0; y < src.cols; y++) 
		{
			if ((src.at<Vec3b>(x, y)[0] >= 240) &&
				(src.at<Vec3b>(x, y)[1] >= 240) &&
				(src.at<Vec3b>(x, y)[2] >= 240)) 
			{
				src.at<Vec3b>(x, y)[0] = 0;
				src.at<Vec3b>(x, y)[1] = 0;
				src.at<Vec3b>(x, y)[2] = 0;
			}
		}
	}

	imshow("Black Background Image", src);

	Mat kernel = (Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -12, 1,
		1, 1, 1);

	Mat imgLaplacian;
	Mat sharp = src;
	filter2D(sharp, imgLaplacian, CV_32F, kernel);
	src.convertTo(sharp, CV_32F);

	Mat imgResult = sharp - imgLaplacian;
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

	vector<vector<Point> > contours;
	Mat imgLaplacianGray;
	cvtColor(imgLaplacian, imgLaplacianGray, CV_BGR2GRAY);
	findContours(imgLaplacianGray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(
			imgResult, 
			contours, 
			static_cast<int>(i), 
			Scalar::all((static_cast<int>(i) + 100) % 255), 
			-1);
	}

	imshow("Borders", imgResult);

	waitKey(0);
	return 0;
}