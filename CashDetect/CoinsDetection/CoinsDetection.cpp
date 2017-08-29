#include "A4Detection.h"
#include "CoinsDetection.h"

#include <cmath>
#include <string>

int calcCoinsSum(Mat& src)
{
	//ensure the image height is less than the image width,
	//otherwise the algorithm may not work correctly
	toHorizontalFrame(src);

	//separates the A4-format sheet from the background
	//catched exception are not dealt with but thrown higher
	try
	{
		src = getA4(src);
	}
	catch (exception e)
	{
		throw exception(e);
	}

	//once again ensure the image height is less than the image width,
	//otherwise the algorithm may not work correctly
	toHorizontalFrame(src);

	//convert to a one-channel image
	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);

	//compute the minimal allowed radius of the potential coins
	double min_radius = src_gray.cols / 38;

	vector<Vec3f> circles = getCoins(src_gray, min_radius);

	//making a copy of the image in order not to draw anything on the original
	//image. Because the color of the next coin may be changed by drawing the
	//center, the contours and the value of the current one.
	Mat src_copy;
	src.copyTo(src_copy);

	Mat hsv;
	cvtColor(src_copy, hsv, CV_RGB2HSV);

	int sum = 0;
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		double radius = circles[i][2];

		int value = getCoinValue(hsv, center, radius, min_radius);
		//if value is equal -1 then the potential coin is either not a coin or
		//at least not a known one. Its value cannot be added the sum,
		//otherwise the sum is reduces and a mistake occurs.
		if (value != -1)
		{
			sum += value;
		}

		//draw the center with a green color
		circle(src_copy, center, 1, Scalar(0, 255, 0), 3, 8, 0);
		//draw the contours with a blue color
		circle(src_copy, center, radius, Scalar(255, 0, 0), 5, 8, 0);

		//draw the coin value on the coin center with a red color
		//if value is equal -1 then the potential coin is either not a coin or
		//at least not a known one.
		//In this case not its value will be drawn but "unknown"
		putText(
			src_copy,
			value == -1 ? "unknown" : to_string(value),
			Point(center.x, center.y),
			FONT_HERSHEY_SCRIPT_COMPLEX,
			1.5,
			Scalar(0, 0, 255),
			3);
	}
	//copy the image with all coins and potential coins marked to 
	//the source image
	src = src_copy;

	return sum;
}

vector<Vec3f> getCoins(Mat& src_1C, double min_radius)
{
	//use GaussianBlur to prevent the false circles from being found
	GaussianBlur(src_1C, src_1C, Size(15, 15), 0);

	vector<Vec3f> circles;

	//find the circles that are the potential coins
	//minimal radius of the potential coins is equal min_radius,
	//minimal radius of the potential coins is equal min_radius * 1.75
	//(the value 1.75 was calculated by hand based on the knowledge about the
	//Ukrainian hryvnia coins sizes),
	//the minimal distance between the centers of each two potential coins
	//is equal to min_radius (as the coins may intersect)
	HoughCircles(
		src_1C,
		circles,
		HOUGH_GRADIENT,
		2,
		min_radius,
		50,
		150,
		min_radius,
		min_radius * 1.75);

	return circles;
}

int getCoinValue(Mat& img_hsv, Point center, double radius, double min_radius)
{
	int value = -1;

	//saturation is used to check the color of the potential coins
	//saturation for 10, 25, 50 and 100 will be higher than 75
	//saturation for 1, 2, 5 will be lower than 35
	//if saturation is in range [35, 75] the coin is not verified
	double s_avg = getSaturationAvg(img_hsv, center);

	//the coefficients on which the min_radius value is multiplied in each
	//if-else statement were calculated by hand based on the knowledge about
	//the Ukrainian hryvnia coins sizes
	if (s_avg > 75)
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

double getSaturationAvg(Mat& img_hsv, Point center)
{
	//hsv[1] a one-channel image (saturation channel)  
	double s_avg = 0;
	//sums the saturation value of 5-pixel cross
	for (int i = -4; i <= 4; i++)
	{
		//pass through x-axis
		s_avg += img_hsv.at<Vec3b>(center.y, center.x + i)[1];
		//pass through y-axis
		s_avg += img_hsv.at<Vec3b>(center.y + i, center.x)[1];
	}
	//divided into 9 * 2, where 9 is the number of pixels in a single axis
	//pass through, and 2 is the number of such passes (x-axis, y-axis)
	//notice that the central pixes saturation value is added twice
	s_avg /= 18;

	return s_avg;
}

void toHorizontalFrame(Mat& img)
{
	if (img.rows > img.cols)
	{
		rotate(img, img, ROTATE_90_CLOCKWISE);
	}
}