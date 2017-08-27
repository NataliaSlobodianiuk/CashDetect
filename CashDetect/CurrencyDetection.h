#pragma once
// standart C++ includes
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <math.h>
#include <fstream>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

// Project includes
#include "Transformations.h"
#include "Configs.h"


/*
*  Function for getting cos between two vectors
*  Gets: 3 points representing 2 vectors:
*  vector1: between p1 and p2
*  vector2: between p2 and p3
*  Returns: cos computed by vector multiplication
*/
float angleBetween(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3);


/* Function for verifing rectangles
*  Gets: image, where rectangle supposed to be drawn, 4 points for rectangle
*  Returns: bool value
*  Verifing:
*  - by points -- if points are in image
*  - by cos -- fabs(cos) < 0.01
*  - by contour area
*  - by convexity
*  - by size of sides and digonal
*/
bool verifyRectangle(const cv::Mat& image, const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4);


/*
*  Function reads templates
*  Algo:
*  - read file with filenames of templates
*  - then reads all images in templates directory
*  - push back into result vector
*  - in case templates filename wasn`t opened or dir was empty
*  -- Return: empty vector
*  - else
*  -- Return: vector
*/
std::vector<cv::Mat> getTemplates();


/* Function for rotating image in case if num of rows > cols
*  In case image cols or rows num is > 1200 resize
*  Gets: cv::Mat object
*  Return: cv::Mat object
*/
cv::Mat rotateAndResize(const cv::Mat& image);


/* Main process function: counts currency in image and output image with drawn currency values and sum
*  Gets: image for detecting and recognizing
*  Returns: double sum
*  Changes: drawing image: draws currency contours and value
*  Algoritm:
*  - get templates for matching
*  - detect keypoints and compute descriptors by SIFT feature detector for each template
*  - go throw all templates and compare them to image
*  -- detect keypoints and compute descriptors for image as it will be changed after every detecting
*  -- in case sth was detected
*  --- find homography for detected object and draw black rectangle in this zone(rectangle)
*  --- detect which currency was detected
*  --- add to sum and draw
*  -- continue detecting with the same template
*  -- in case nothing left to detect go to the next template
*  - draw sum at image and output result
*/
double getCurrencySum(const cv::Mat& image, cv::Mat& drawing);


