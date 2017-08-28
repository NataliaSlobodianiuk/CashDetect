#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

/**
	Finds the A4-format sheet of paper on the image and separates it from 
	the background.

	@param src a BGR-image on which the A4-format sheet should be found and
	separated from the background. If not the exception is thrown.

	@returns the separated from the background A4-format sheet of paper.
**/
Mat getA4(Mat& src);

/**
	Transforms the image separating the rectangle definited by its vertices
	from the background.
	At first finds the transformed verteces of the rectangle (what their
	coords should be on the transformed image).
	Then uses getPerspectiveTransform and warpPerspective to make a 
	transformation.

	@param src a BGR-image on which the A4-format sheet was found and should
	separated from the background.
**/
void fourVertecesTransform(Mat& src, vector<Point2f> verteces);

/**
	Sorts the verteces of the rectangle in the following order:
	[0] - top left;
	[1] - top right;
	[2] - bottom left;
	[3] - bottom right.

	@param verteces a vector of 4 points which are the verteces of the
	rectangle. They are already sorted (order by descending) by y coordinate.
**/
void sortVerteces(vector<Point2f>& verteces);