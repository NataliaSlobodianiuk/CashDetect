#include <iostream>
#include <cmath>
#include <string>
#include <ctime>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int getCoinValue(Mat& img, Point center, double radius, double min_radius);
double getBGRDifference(Mat& img, Point center);
vector<Vec3f> getCircles(Mat& src_1C, double min_radius);

void sortVerteces(vector<Point2f>& verteces);
void fourVertecesTransform(Mat& src, vector<Point2f> verteces);

int getCoinsSum(Mat& src);
Mat getA4(Mat& src);

int kernel3x3[] = {
	1, 1, 1,
	1, 1, 1,
	1, 1, 1
};

int kernel5x5[] = {
	2, 0, 2, 0, 2,
	0, 2, 0, 2, 0,
	2, 0, 2, 0, 2,
	0, 2, 0, 2, 0,
	2, 0, 2, 0, 2,
};

Mat kernel = Mat(5, 5, CV_8UC1, kernel5x5);

int main(int argc, char **argv)
{
	Mat src;
	Mat src_gray, src_gray_copy;
	Mat mask;
	vector<vector<Point>> contours;

	for (int i = 1; i < argc; i++)
	{
		src = imread(argv[i]);

		if (!src.data)
			return -1;

		int sum = 0;
		clock_t begin = clock();
		{
			int height = src.rows;
			int width = src.cols;

			if (height > width)
			{
				rotate(src, src, ROTATE_90_CLOCKWISE);
			}
			namedWindow("Source", CV_WINDOW_FREERATIO);
			imshow("Source", src);

			src = getA4(src);
			height = src.rows;
			width = src.cols;

			if (height > width)
			{
				rotate(src, src, ROTATE_90_CLOCKWISE);
			}
			namedWindow("SourceA4", CV_WINDOW_FREERATIO);
			imshow("SourceA4", src);

			sum = getCoinsSum(src);
			namedWindow("Result", CV_WINDOW_FREERATIO);
			imshow("Result", src);
		}
		clock_t end = clock();

		cout << "Time elapsed: " << (double)(end - begin) / CLOCKS_PER_SEC << " second(s)" << endl;

		cout << "Sum: " << sum / 100 << " hryvnia(s) " << sum % 100 << " hryvnia coins." << endl;

		int key = waitKey(0);

		if (key == 27)
		{
			return 0;
		}
	}

	return 0;
}

int getCoinValue(Mat& img, Point center, double radius, double min_radius)
{
	int value = -1;

	double bgr_difference = getBGRDifference(img, center);

	if (radius > min_radius && radius < min_radius * 1.1)
	{
		if (bgr_difference > 10)
		{
			value = 10;
		}
		else
		{
			value = 1;
		}
	}
	else if (radius < min_radius * 1.15 && bgr_difference < 5)
	{
		value = 2;
	}
	else if (radius > min_radius * 1.35 && 
		radius < min_radius * 1.45 && 
		bgr_difference > 10)
	{
		value = 25;
	}
	else if (radius > min_radius * 1.5 && radius < min_radius * 1.6)
	{
		if (bgr_difference > 10)
		{
			value = 50;
		}
		else
		{
			value = 5;
		}
	}
	else if (radius > min_radius * 1.68 &&
		radius < min_radius * 1.72 && 
		bgr_difference > 10)
	{
		value = 100;
	}

	return value;
}

double getBGRDifference(Mat& img, Point center)
{
	double bgr_difference = 0;

	for (int i = -3; i <= 3; i++)
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

	bgr_difference /= 13;

	return bgr_difference;
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
		min_radius * 1.5,
		30,
		50,
		min_radius,
		min_radius * 2);

	return circles;
}

int getCoinsSum(Mat& src)
{
	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	namedWindow("Gray", CV_WINDOW_FREERATIO);
	imshow("Gray", src_gray);

	double min_radius = (double)max<int>(src_gray.cols, src_gray.rows) / 40;

	vector<Vec3f> circles = getCircles(src_gray, min_radius);

	cout << "Number of coins: " << circles.size() << ".\n";

	int sum = 0;
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		int value = getCoinValue(src, center, radius, min_radius);
		if (value != -1)
		{
			sum += value;
		}

		circle(src, center, 1, Scalar(0, 255, 0), 3, 8, 0);
		circle(src, center, radius, Scalar(255, 0, 0), 5, 8, 0);

		putText(
			src,
			value == -1 ? "unknown coin" : to_string(value),
			Point(center.x - radius, center.y - radius),
			FONT_HERSHEY_SCRIPT_COMPLEX,
			3,
			Scalar(0, 0, 255),
			3);
	}
	return sum;
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

	Mat kernel(Mat::ones(Size(5, 5), CV_8UC1));
	filter2D(mask, mask, CV_8UC1, kernel);
	namedWindow("Mask After Filter2D", CV_WINDOW_FREERATIO);
	imshow("Mask After Filter2D", mask);

	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	int j = -1;
	for (auto cnt : contours)
	{
		j++;

		double arcL = arcLength(cnt, true);

		if (arcL < ((src.cols + src.rows) / 2))
		{
			drawContours(mask, contours, j, Scalar(0), -1);
			continue;
		}

		approxPolyDP(cnt, cnt, 0.05 * arcL, true);

		if (cnt.size() == 4)
		{
			drawContours(mask, contours, j, Scalar(255), -1);

			vector<Point2f> verteces;
			verteces.push_back(cnt[0]);
			verteces.push_back(cnt[1]);
			verteces.push_back(cnt[2]);
			verteces.push_back(cnt[3]);

			fourVertecesTransform(A4, verteces);
			break;
		}
	}

	namedWindow("Mask After Filling Contours", CV_WINDOW_FREERATIO);
	imshow("Mask After Filling Contours", mask);

	return A4;
}