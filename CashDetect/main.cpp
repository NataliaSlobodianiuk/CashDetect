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

	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	GaussianBlur(src_gray, src_gray, Size(3, 3), 0, 0);

	double min_radius = src.cols / 50;
	vector<Vec3f> circles;
	HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1, min_radius * 2, 100, 65, min_radius);

	cout << circles.size() << endl;

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		circle(src, center, 1, Scalar(0, 255, 0), -1, 8, 0);
		circle(src, center, radius, Scalar(0, 0, 255), 1, 8, 0);
	}

	imshow("Result", src);

	waitKey(0);
	return 0;
}