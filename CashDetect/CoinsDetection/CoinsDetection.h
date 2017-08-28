#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

/**
	Calculates the sum of the coins found on the image.
	At first looks for an A4 sheet. If finds starts looking for the coins
	on the A4-format sheet, determining their values and adding them.
	If not throws an exception.

	@param src a BGR-image where there is (not) an A4-format sheet of paper
	on which the coins may (not) be located.

	@returs the sum of the coins found on the image.
**/
int calcCoinsSum(Mat& src);

/**
	Finds the potential coins based on their allowed radius.
	HoughCircles is used to find the potential coins on the image.
	The circle is said to be a potential coin just in case its radius is 
	in range [min_radius, max_radius].

	@param src_1C a one-channel image on which the potential coins
	may (not) be located.
	@param min_radius value to determine what is the minimal allowed radius
	of the coins on the picture (from this value the maximum allowed radius
	is computed). These values are user to form the range of the potential 
	coins allowed radius [min_radius, max_radius].

	@returns the vector of potential coins.
**/
vector<Vec3f> getCoins(Mat& src_1C, double min_radius);

/**
	Determines the coin value based on its radius and color.
	getSaturationAvg is used to check is the color of the potential coin.
	The coin value is found just in case both the color and the radius
	of the potential coin matches any known coins. 

	@param img a BGR-image where the potential coin is located.
	@param center point to determine where is the center of potential coin.
	@param radius value to determine what is the radius of the potential coin.
	@param min_radius value to determine what is the minimal allowed radius
	of the coins on the picture (from this value the maximum allowed radius
	is computed).
	
	@returns the value of the coin if the match was found, else -1.
**/
int getCoinValue(Mat& img, Point center, double radius, double min_radius);

/**
	Computes the average saturation value of 17 pixels (a 5-pixel cross).

	@param img a BGR-image to be conterted to hsv format, its saturation
	values are user to compute the the average saturation value of 17 pixels.
	@param center point to determine where is the center of the 5-pixel cross.

	@returns the average saturation value of 17 pixels (a 5-pixel cross).
**/
double getSaturationAvg(Mat& img, Point center);

/**
	Rotates the image if its height is bigger than its width.

	@param img an image of any format to be rotated if its number of rows is 
	bigger than its number of cols 
**/
void toHorizontalFrame(Mat& img);
