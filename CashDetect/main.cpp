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
Mat getMask(Mat& src1, Mat& src2);

int detectCoins(Mat& src, Mat& src_gray);

int main(int argc, char **argv)
{
	Mat src;
	Mat src_gray;
	Mat src_hsv;
	vector<Mat> bgr_channels, hsv_channels, mask_channels;
	Mat src_h, src_s, src_v;
	Mat mask;

	int kernel3x3[] = {
		2, 0, 2, 
		0, 2, 0,
		2, 0, 2
	};

	int kernel5x5[] = { 
		2, 0, 2, 0, 2,
		0, 2, 0, 2, 0,
		2, 0, 2, 0, 2,
		0, 2, 0, 2, 0,
		2, 0, 2, 0, 2,
	};

	Mat kernel(5, 5, CV_8U, kernel5x5);

	for (int i = 1; i < argc; i++)
	{
		src = imread(argv[i]);

		if (!src.data)
			return -1;

		int sum = 0;
		clock_t begin = clock();
		{
			resize(src, src, Size(src.cols / 3, src.rows / 3));

			//cvtColor(src, src, CV_BGR2Lab);
			//split(src, bgr_channels);
			//equalizeHist(bgr_channels[0], bgr_channels[0]);
			//equalizeHist(bgr_channels[1], bgr_channels[1]);
			//equalizeHist(bgr_channels[2], bgr_channels[2]);
			//merge(bgr_channels, src);
			//cvtColor(src, src, CV_Lab2BGR);

			namedWindow("Source", CV_WINDOW_FREERATIO);
			imshow("Source", src);

			cvtColor(src, src_hsv, COLOR_BGR2HSV);

			namedWindow("HSV", CV_WINDOW_FREERATIO);
			imshow("HSV", src_hsv);

			split(src_hsv, hsv_channels);
			src_h = hsv_channels[0];
			src_s = hsv_channels[1];
			src_v = hsv_channels[2];

			equalizeHist(src_h, src_h);
			equalizeHist(src_s, src_s);
			equalizeHist(src_v, src_v);

			namedWindow("Hue", CV_WINDOW_FREERATIO);
			imshow("Hue", src_h);
			namedWindow("Saturation", CV_WINDOW_FREERATIO);
			imshow("Saturation", src_s);
			namedWindow("Value", CV_WINDOW_FREERATIO);
			imshow("Value", src_v);

			mask_channels.push_back(src_s);
			mask_channels.push_back(src_v);

			mask = getMask(mask_channels[0], mask_channels[1]);
			bitwise_not(mask, mask);

			medianBlur(mask, mask, 9);
			erode(mask, mask, kernel);
			dilate(mask, mask, kernel);
			namedWindow("Mask", CV_WINDOW_FREERATIO);
			imshow("Mask", mask);

			cvtColor(src, src_gray, CV_BGR2GRAY);
			bitwise_and(src_gray, mask, src_gray);
			namedWindow("Gray", CV_WINDOW_FREERATIO);
			imshow("Gray", src_gray);

			resize(src_gray, src_gray, Size(src_gray.cols * 3, src_gray.rows * 3));
			resize(src, src, Size(src.cols * 3, src.rows * 3));

			sum = detectCoins(src, src_gray);
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

Mat getMask(Mat& src1, Mat& src2)
{
	if (src1.cols != src2.cols || src1.rows != src2.rows)
	{
		throw exception("Wrong formats.");
	}

	Mat mask(src1.rows, src1.cols, CV_8UC1);

	for (int i = 0; i < mask.cols; i++)
	{
		for (int j = 0; j < mask.rows; j++)
		{
			mask.at<uchar>(Point(i, j)) = (
				abs(src1.at<uchar>(Point(i, j)) - src2.at<uchar>(Point(i, j))) > 225 ?
				0 : 
				255);
		}
	}

	return mask;
}

int detectCoins(Mat& src, Mat& src_1C)
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

		circle(src, center, 1, Scalar(0, 255, 0), 5, 8, 0);
		circle(src, center, radius, Scalar(255, 0, 0), 9, 8, 0);

		putText(
			src,
			value == -1 ? "unknown coin": to_string(value),
			Point(center.x - radius, center.y - radius),
			FONT_HERSHEY_SCRIPT_COMPLEX,
			3,
			Scalar(0, 0, 255),
			3);
	}
	return sum;
}