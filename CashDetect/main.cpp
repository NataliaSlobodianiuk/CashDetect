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

void transformFourVerteces(Mat& src, vector<Point> verteces);

int getCoinsSum(Mat& src, Mat& src_gray);
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

Mat kernel = Mat(3, 3, CV_8UC1, kernel3x3);


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
			resize(src, src, Size(src.cols / 3, src.rows / 3));

			namedWindow("Source", CV_WINDOW_FREERATIO);
			imshow("Source", src);

			src = getA4(src);
			namedWindow("SourceA4", CV_WINDOW_FREERATIO);
			imshow("SourceA4", src);

			cvtColor(src, src_gray, CV_BGR2GRAY);
			namedWindow("Gray", CV_WINDOW_FREERATIO);
			imshow("Gray", src_gray);

			src_gray.copyTo(src_gray_copy);

			GaussianBlur(src_gray_copy, src_gray_copy, Size(15, 15), 0);
			adaptiveThreshold(src_gray_copy, mask, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 15, 1);
			namedWindow("Mask After Adaptive", CV_WINDOW_FREERATIO);
			imshow("Mask After Adaptive", mask);

			medianBlur(mask, mask, 3);
			medianBlur(mask, mask, 5);
			namedWindow("Mask After Median Blur", CV_WINDOW_FREERATIO);
			imshow("Mask After Median Blur", mask);

			morphologyEx(mask, mask, MORPH_ERODE, kernel, Point(-1, -1), 1);
			morphologyEx(mask, mask, MORPH_DILATE, kernel, Point(-1, -1), 2);
			namedWindow("Mask After Morphology", CV_WINDOW_FREERATIO);
			imshow("Mask After Morphology", mask);

			findContours(mask, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
			int j = -1;
			for (auto cnt : contours)
			{
				j++;

				double area = contourArea(cnt);

				if (area > 22500)
				{
					continue;
				}
				if (area < 1600 || cnt.size() < 5)
				{
					drawContours(mask, contours, j, Scalar(0), -1);
					continue;
				}

				drawContours(mask, contours, j, Scalar(255), -1);
			}
			namedWindow("Contours", CV_WINDOW_FREERATIO);
			imshow("Contours", mask);

			bitwise_and(src_gray, mask, src_gray);
			namedWindow("Gray With Mask", CV_WINDOW_FREERATIO);
			imshow("Gray With Mask", src_gray);

			sum = getCoinsSum(src, src_gray);
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

	if (radius > min_radius && radius < min_radius * 1.5)
	{
		if (bgr_difference > 30)
		{
			value = 10;
		}
		else
		{
			value = 2;
		}
	}
	else if (radius < min_radius * 1.8 && bgr_difference > 30)
	{
		value = 25;
	}
	else if (radius < min_radius * 2)
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
	else if (radius < min_radius * 2.3 && bgr_difference > 30)
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
	GaussianBlur(src_1C, src_1C, Size(13, 13), 0);

	vector<Vec3f> circles;

	HoughCircles(
		src_1C,
		circles,
		HOUGH_GRADIENT,
		1,
		min_radius,
		30,
		30,
		min_radius,
		min_radius * 2);

	return circles;
}

int getCoinsSum(Mat& src, Mat& src_1C)
{
	double min_radius = (double)max<int>(src_1C.cols, src_1C.rows) / 50;

	vector<Vec3f> circles = getCircles(src_1C, min_radius);

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

		circle(src, center, 1, Scalar(0, 255, 0), 1, 8, 0);
		circle(src, center, radius, Scalar(255, 0, 0), 3, 8, 0);

		putText(
			src,
			value == -1 ? "unknown coin" : to_string(value),
			Point(center.x - radius, center.y - radius),
			FONT_HERSHEY_SCRIPT_COMPLEX,
			1,
			Scalar(0, 0, 255),
			1);
	}
	return sum;
}

void transformFourVerteces(Mat& src, vector<Point> verteces)
{
	/*ToDo*/
}

Mat getA4(Mat& src)
{
	Mat A4 = src;
	Mat gray, mask;
	vector<vector<Point>> contours;
	vector<Mat> rgb_channels;

	cvtColor(src, gray, CV_BGR2GRAY);
	GaussianBlur(gray, gray, Size(5, 5), 0);

	Canny(gray, mask, 75, 200);

	Mat kernel(Mat::ones(Size(9, 9), CV_8UC1));
	filter2D(mask, mask, CV_8UC1, kernel);

	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	int j = -1;
	for (auto cnt : contours)
	{
		j++;

		double arcL = arcLength(cnt, true);
		approxPolyDP(cnt, cnt, 0.05 * arcL, true);

		drawContours(mask, contours, j, Scalar(255), -1);
		
		if (cnt.size() == 4)
		{
			break;
		}
	}

	split(src, rgb_channels);
	bitwise_and(rgb_channels[0], mask, rgb_channels[0]);
	bitwise_and(rgb_channels[1], mask, rgb_channels[1]);
	bitwise_and(rgb_channels[2], mask, rgb_channels[2]);
	merge(rgb_channels, A4);

	return A4;
}