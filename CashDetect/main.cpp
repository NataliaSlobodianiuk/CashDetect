#include <iostream>
#include <cmath>
#include <string>
#include <ctime>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int getCoinValue(Mat& img, Point center, double radius);
double getBGRDifference(Mat& img, Point center);

int main(int argc, char **argv)
{
	for (int i = 1; i < argc; i++)
	{
		Mat src = imread(argv[i]);

		if (!src.data)
			return -1;

		clock_t begin = clock();

		resize(src, src, Size(3600, 2160));

		Mat src_gray;
		cvtColor(src, src_gray, COLOR_BGR2GRAY);

		medianBlur(src_gray, src_gray, 7);

		double min_radius = (double)max<int>(src.cols, src.rows) / 50;
		vector<Vec3f> circles;
		HoughCircles(src_gray, circles, HOUGH_GRADIENT, 3, min_radius * 1.5, 65, 150, min_radius, min_radius * 2);

		cout << "Number of coins: " << circles.size() << ".\n";

		int sum = 0;
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);

			int value = getCoinValue(src, center, radius);
			sum += value;

			circle(src, center, 1, Scalar(0, 255, 0), 5, 8, 0);
			circle(src, center, radius, Scalar(0, 0, 255), 10, 8, 0);

			putText(src, to_string(value), Point(center.x - radius, center.y - radius), FONT_HERSHEY_SCRIPT_COMPLEX, 3, Scalar::all(0), 3);
		}

		clock_t end = clock();

		cout << "Sum: " << sum / 100 << " hryvnia(s) " << sum % 100 << " hryvnia coins." << endl;
		cout << "Time elapsed: " << (double)(end - begin) / CLOCKS_PER_SEC << " second(s)" << endl;

		namedWindow("Result", CV_WINDOW_KEEPRATIO);
		imshow("Result", src);

		int key = waitKey(0);

		if (key == 27)
		{
			return 0;
		}
	}

	return 0;
}

int getCoinValue(Mat& img, Point center, double radius)
{
	int value = -1;

	double bgr_difference = getBGRDifference(img, center);
	
	if (radius < 110)
	{
		value = 10;
	}
	else if (radius >= 110 && radius < 125)
	{
		value = 25;
	}
	else if (radius >= 125 && radius < 145)
	{
		if (bgr_difference > 30)
		{
			value = 50;
		}
		else
		{
			value = 5;
		}
	}
	else
	{
		value = 100;
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