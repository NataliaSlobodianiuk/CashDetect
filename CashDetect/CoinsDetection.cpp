#include "CoinsDetection.h"

#include <cmath>
#include <string>

int getCoinsSum(Mat& src)
{
	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	namedWindow("Gray", CV_WINDOW_FREERATIO);
	imshow("Gray", src_gray);

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

Mat getA4(Mat& src)
{
	Mat A4 = src;
	Mat gray, mask;
	vector<vector<Point>> contours;

	cvtColor(src, gray, CV_BGR2GRAY);
	GaussianBlur(gray, gray, Size(15, 15), 0);

	Canny(gray, mask, 10, 30);
	namedWindow("Mask After Canny", CV_WINDOW_FREERATIO);
	imshow("Mask After Canny", mask);

	Mat kernel(Mat::ones(Size(9, 9), CV_8UC1));
	filter2D(mask, mask, CV_8UC1, kernel);
	namedWindow("Mask After Filter2D", CV_WINDOW_FREERATIO);
	imshow("Mask After Filter2D", mask);

	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	int maxLengthCntNum = -1;
	double maxLength = -1;
	int j = -1;
	for (auto cnt : contours)
	{
		j++;

		double arcL = arcLength(cnt, true);

		if (arcL < ((src.cols + src.rows) / 10))
		{
			drawContours(mask, contours, j, Scalar(0), -1);
			continue;
		}

		if (arcL > maxLength)
		{
			maxLength = arcL;
			maxLengthCntNum = j;
		}
	}

	vector<Point> longestCnt = contours[maxLengthCntNum];
	approxPolyDP(longestCnt, longestCnt, 0.05 * maxLength, true);

	if (longestCnt.size() == 4)
	{
		vector<Point2f> verteces;
		verteces.push_back(longestCnt[0]);
		verteces.push_back(longestCnt[1]);
		verteces.push_back(longestCnt[2]);
		verteces.push_back(longestCnt[3]);

		fourVertecesTransform(A4, verteces);
	}
	else
	{
		char* eMessage = "Wrong input: Cannot detect A4 on a photo.";
		throw exception(eMessage);
	}

	namedWindow("Mask After Filling Contours", CV_WINDOW_FREERATIO);
	imshow("Mask After Filling Contours", mask);

	return A4;
}

void fourVertecesTransform(Mat& src, vector<Point2f> verteces)
{
	sortVerteces(verteces);

	double length01 = sqrt(
		(verteces[0].x - verteces[1].x) * (verteces[0].x - verteces[1].x) +
		(verteces[0].y - verteces[1].y) * (verteces[0].y - verteces[1].y));
	double length23 = sqrt(
		(verteces[2].x - verteces[3].x) * (verteces[2].x - verteces[3].x) +
		(verteces[2].y - verteces[3].y) * (verteces[2].y - verteces[3].y));
	double maxWidth = max(length01, length23);

	double length02 = sqrt(
		(verteces[0].x - verteces[2].x) * (verteces[0].x - verteces[2].x) +
		(verteces[0].y - verteces[2].y) * (verteces[0].y - verteces[2].y));
	double length13 = sqrt(
		(verteces[1].x - verteces[3].x) * (verteces[1].x - verteces[3].x) +
		(verteces[1].y - verteces[3].y) * (verteces[1].y - verteces[3].y));
	double maxHeight = max(length02, length13);

	vector<Point2f> transformedVerteces;
	if (maxWidth > maxHeight)
	{
		transformedVerteces = {
			Point2f(0, 0),
			Point2f(2700, 0),
			Point2f(0, 1928),
			Point2f(2700, 1928)
		};
	}
	else
	{
		transformedVerteces = {
			Point2f(0, 0),
			Point2f(0, 1928),
			Point2f(2700, 1928),
			Point2f(2700, 0)
		};
	}

	Mat transformationMatrix = getPerspectiveTransform(verteces, transformedVerteces);
	warpPerspective(src, src, transformationMatrix, Size(2700, 1928));
}

void sortVerteces(vector<Point2f>& verteces)
{
	if (verteces[0].x > verteces[1].x)
	{
		swap(verteces[0], verteces[1]);
	}

	if (verteces[2].x > verteces[3].x)
	{
		swap(verteces[2], verteces[3]);
	}
}
