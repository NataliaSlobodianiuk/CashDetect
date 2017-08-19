#include <iostream>
#include <cmath>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int getCoinValue(Mat& img, Point center, double radius);
double getBGRDifference(Mat& img, Point center);

int main(int argc, char **argv)
{
	Mat src = imread(argv[1]);

	if (!src.data)
		return -1;

	//namedWindow("Source Image", CV_WINDOW_KEEPRATIO);
	//imshow("Source Image", src);

	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	GaussianBlur(src_gray, src_gray, Size(5, 5), 0, 0);

	double min_radius = (double)max<int>(src.cols, src.rows) / 50;
	vector<Vec3f> circles;
	HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1, min_radius * 1.5, 20, 45, min_radius, min_radius * 2);

	cout << "Number of coins: " << circles.size() << ".\n";

	int sum = 0;
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		int value = getCoinValue(src, center, radius);
		sum += value;

		circle(src, center, 1, Scalar(0, 255, 0), -1, 8, 0);
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);

		putText(src, to_string(value), Point(center.x - radius, center.y - radius), FONT_HERSHEY_DUPLEX, 1, Scalar::all(255));
	}
	cout << "Sum: " << sum / 100 << " hryvnia(s) " << sum % 100 << " hryvnia coins." << endl;

	namedWindow("Result", CV_WINDOW_KEEPRATIO);
	imshow("Result", src);

	waitKey(0);
	return 0;
}

int getCoinValue(Mat& img, Point center, double radius)
{
	int value = -1;

	if (radius < 100)
	{
		value = 10;
	}
	else if (radius >= 100 && radius < 130)
	{
		value = 25;
	}
	else
	{
		double bgr_difference = getBGRDifference(img, center);

		if (bgr_difference > 40)
		{
			value = 50;
		}
		else
		{
			value = 5;
		}
	}

	return value;
}

double getBGRDifference(Mat& img, Point center)
{
	double bgr_difference = 0;

	for (int i = -2; i <= 2; i++)
	{
		Vec3b x_axis = img.at<Vec3b>(Point(center.x + i, center.y));
		bgr_difference += max<uchar>(
			max<uchar>(abs(x_axis[0] - x_axis[1]), abs(x_axis[0] - x_axis[2])),
			abs(x_axis[1] - x_axis[2])); 
		Vec3b y_axis = img.at<Vec3b>(Point(center.x, center.y + i));
		bgr_difference += max<uchar>(
			max<uchar>(abs(y_axis[0] - y_axis[1]), abs(y_axis[0] - y_axis[2])),
			abs(y_axis[1] - y_axis[2]));
	}

	bgr_difference /= 10;

	return bgr_difference;
}