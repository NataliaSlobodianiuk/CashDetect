#pragma once
#include<vector>

#include"opencv2\opencv.hpp"


class Cash
{
	// Image for detected cash
	cv::Mat image;
	// Point for drawing rotated rectangle around image 
	std::vector<cv::Point2f> corners;
	// Value
	int value;

	// Bool value for knowing if value was detected
	bool detected = false;

public:
	// Basic constructor - set image
	Cash(cv::Mat _image);
	// Return basic info
	int detectedValue();
	// 
	Cash();
	~Cash();
};

