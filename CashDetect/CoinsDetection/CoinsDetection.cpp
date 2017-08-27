#include "A4Detection.h"
#include "CoinsDetection.h"

#include <cmath>
#include <string>

int getCoinsSum(Mat& src)
{
	int height = src.rows;
	int width = src.cols;

	if (height > width)
	{
		rotate(src, src, ROTATE_90_CLOCKWISE);
	}

	try
	{
		src = getA4(src);
	}
	catch (exception e)
	{
		throw exception(e);
	}

	height = src.rows;
	width = src.cols;

	if (height > width)
	{
		rotate(src, src, ROTATE_90_CLOCKWISE);
	}

	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);

	double min_radius = (double)max<int>(src_gray.cols, src_gray.rows) / 38;

	vector<Vec3f> circles = getCircles(src_gray, min_radius);

	Mat src_copy;
	src.copyTo(src_copy);

	int sum = 0;
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		double radius = circles[i][2];

		int value = getCoinValue(src, center, radius, min_radius);
		if (value != -1)
		{
			sum += value;
		}

		circle(src_copy, center, 1, Scalar(0, 255, 0), 3, 8, 0);
		circle(src_copy, center, radius, Scalar(255, 0, 0), 5, 8, 0);

		putText(
			src_copy,
			value == -1 ? "unknown" : to_string(value),
			Point(center.x, center.y),
			FONT_HERSHEY_SCRIPT_COMPLEX,
			1.5,
			Scalar(0, 0, 255),
			3);
	}
	src = src_copy;

	return sum;
}

vector<Vec3f> getCircles(Mat& src_1C, double min_radius)
{
	GaussianBlur(src_1C, src_1C, Size(15, 15), 0);

	vector<Vec3f> circles;

	HoughCircles(
		src_1C,
		circles,
		HOUGH_GRADIENT,
		1,
		min_radius,
		30,
		50,
		min_radius,
		min_radius * 1.75);

	return circles;
}

int getCoinValue(Mat& img, Point center, double radius, double min_radius)
{
	int value = -1;

	double s_avg = getSaturationAvg(img, center);

	//saturation for 10, 25, 50 and 100 will be higher than 70
	//saturation for 1, 2, 5 will be lower than 35
	//if saturation is in range [35, 70] the coin is not verified
	if (s_avg > 70)
	{
		if (radius >= min_radius && radius < min_radius * 1.1)
		{
			value = 10;
		}
		else if (radius >= min_radius * 1.27 && radius <= min_radius * 1.4)
		{
			value = 25;
		}
		else if (radius >= min_radius * 1.43 && radius <= min_radius * 1.57)
		{
			value = 50;
		}
		else if (radius >= min_radius * 1.58 && radius <= min_radius * 1.75)
		{
			value = 100;
		}
	}
	else if (s_avg < 35)
	{
		if (radius >= min_radius && radius < min_radius * 1.1)
		{
			value = 1;
		}
		else if (radius <= min_radius * 1.15)
		{
			value = 2;
		}
		else if (radius >= min_radius * 1.43 && radius <= min_radius * 1.57)
		{
			value = 5;
		}
	}

	return value;
}

double getSaturationAvg(Mat& img, Point center)
{
	Mat hsv;
	cvtColor(img, hsv, CV_RGB2HSV);

	double s_avg = 0;

	for (int i = -5; i <= 5; i++)
	{
		s_avg += hsv.at<Vec3b>(center.y, center.x + i)[1];
		s_avg += hsv.at<Vec3b>(center.y + i, center.x)[1];
	}

	s_avg /= 21;

	return s_avg;
}
