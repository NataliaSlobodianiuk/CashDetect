#include "A4Detection.h"

Mat getA4(Mat& src)
{
	Mat A4 = src;
	Mat gray, mask;
	vector<vector<Point>> contours;

	//convert to a one-channel image
	cvtColor(src, gray, CV_BGR2GRAY);

	//use GaussianBlur to prevent the false contours from being found
	GaussianBlur(gray, gray, Size(15, 15), 0);

	Canny(gray, mask, 10, 30);

	//2D filter with size 9x9 is used to make thin lines on the mask thicker
	//and make the contours closed
	Mat kernel(Mat::ones(Size(9, 9), CV_8UC1));
	filter2D(mask, mask, CV_8UC1, kernel);

	//find just external contours (the longest one should be A4-format sheet
	//contour, the rest are useless blurs and etc.
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//a cycle to find the longest contour
	int maxLengthCntNum = -1;
	double maxLength = -1;
	int j = -1;
	for (auto cnt : contours)
	{
		j++;

		//find a current countour perimeter
		double cContPerimeter = arcLength(cnt, true);

		//the contours with too short perimeters are filled with a black color
		//and ommited for further calculations
		if (cContPerimeter < ((src.cols + src.rows) / 10))
		{
			drawContours(mask, contours, j, Scalar(0), -1);
			continue;
		}

		if (cContPerimeter > maxLength)
		{
			maxLength = cContPerimeter;
			maxLengthCntNum = j;
		}
	}

	vector<Point> longestCnt = contours[maxLengthCntNum];
	//approximate the longest contour, ellipson is a 5% multiplied by the
	//contour length
	approxPolyDP(longestCnt, longestCnt, 0.05 * maxLength, true);

	//the A4-formal sheet contour should be approximated to 4 points
	//(4 verteces of the rectangle), if not the exception is thrown
	if (longestCnt.size() == 4)
	{
		vector<Point2f> verteces;
		verteces.push_back(longestCnt[0]);
		verteces.push_back(longestCnt[1]);
		verteces.push_back(longestCnt[2]);
		verteces.push_back(longestCnt[3]);

		//separate A4-format sheet from the background
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
	//verteces or the rectangle shoulde be sorted in the following order:
	//[0] - top left;
	//[1] - top right;
	//[2] - bottom left;
	//[3] - bottom right.
	sortVerteces(verteces);

	//find the longest width
	double length01 = sqrt(
		(verteces[0].x - verteces[1].x) * (verteces[0].x - verteces[1].x) +
		(verteces[0].y - verteces[1].y) * (verteces[0].y - verteces[1].y));
	double length23 = sqrt(
		(verteces[2].x - verteces[3].x) * (verteces[2].x - verteces[3].x) +
		(verteces[2].y - verteces[3].y) * (verteces[2].y - verteces[3].y));
	double maxWidth = max(length01, length23);

	//find the longest height
	double length02 = sqrt(
		(verteces[0].x - verteces[2].x) * (verteces[0].x - verteces[2].x) +
		(verteces[0].y - verteces[2].y) * (verteces[0].y - verteces[2].y));
	double length13 = sqrt(
		(verteces[1].x - verteces[3].x) * (verteces[1].x - verteces[3].x) +
		(verteces[1].y - verteces[3].y) * (verteces[1].y - verteces[3].y));
	double maxHeight = max(length02, length13);

	//if the width is bigger than the height, the rectangle is horizontal 
	//otherwise it is vertical
	//transformedVerteces should be chosen correctly in order not to 
	//flatten/stretch the image
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

	//transform the source image by separating the rectangle from the 
	//background
	//the coeficient 2700, 1928 were chosen based on the knowledge about the
	//rectangle (the source image is being transformed to) being the A4-format
	//sheet of paper (1: sqrt(2));
	Mat transformationMatrix = 
		getPerspectiveTransform(verteces, transformedVerteces);
	warpPerspective(src, src, transformationMatrix, Size(2700, 1928));
}

void sortVerteces(vector<Point2f>& verteces)
{
	//[0] - top left - is min(x) from first and second min(y);
	//[1] - top right - is max(x) from first and second min(y);
	if (verteces[0].x > verteces[1].x)
	{
		swap(verteces[0], verteces[1]);
	}

	//[2] - bottom left - is min(x) from third and fourth min(y);
	//[3] - bottom right - is max(x) from third and fourth min(y).
	if (verteces[2].x > verteces[3].x)
	{
		swap(verteces[2], verteces[3]);
	}
}
