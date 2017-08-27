#include "A4Detection.h"

Mat getA4(Mat& src)
{
	Mat A4 = src;
	Mat gray, mask;
	vector<vector<Point>> contours;

	cvtColor(src, gray, CV_BGR2GRAY);
	GaussianBlur(gray, gray, Size(15, 15), 0);

	Canny(gray, mask, 10, 30);

	Mat kernel(Mat::ones(Size(9, 9), CV_8UC1));
	filter2D(mask, mask, CV_8UC1, kernel);

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
