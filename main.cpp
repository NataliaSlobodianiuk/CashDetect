#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include<math.h>

#include"opencv2\opencv.hpp"
#include"opencv2\core.hpp"
#include"opencv2\imgproc.hpp"
#include"opencv2\xfeatures2d.hpp"
#include"opencv2\flann.hpp"

#include "CLI.h"

float angleBetween(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3)
{
	// Compute lengths
	double len1 = sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y));
	double len2 = sqrt((p3.x - p2.x)*(p3.x - p2.x) + (p3.y - p2.y)*(p3.y - p2.y));

	// Compute dots
	double dot = sqrt(fabs((p2.x - p1.x)*(p3.x - p2.x)) + fabs((p2.y - p1.y)*(p3.y - p2.y)));

	// Compute cos
	double a = dot / (len1 * len2);
	// Return
	return a;
}

bool verifyRectangle(const cv::Mat& image ,const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4)
{
	// Verify points
	if (p1.x >= 0 && p1.x < image.cols && p1.y >= 0 && p1.y < image.rows && p2.x >= 0 && p2.x < image.cols && p2.y >= 0 && p2.y < image.cols
		&& p3.x >= 0 && p3.x < image.cols && p3.y >= 0 && p3.y < image.cols && p4.x >= 0 && p4.x < image.cols && p4.y >= 0 && p4.y < image.cols) {
		if ((fabs(angleBetween(p1, p2, p3)) < 0.005) && (fabs(angleBetween(p2, p3, p4)) < 0.005) && (fabs(angleBetween(p3, p4, p1)) < 0.005) && (fabs(angleBetween(p4, p1, p2)) < 0.005))
		{
			return true;
		}
		else {
			return false;
		}
	}
	else {
		return false;
	}
}

// Function for extracting cash
std::vector<cv::Mat> extractCash(const cv::Mat& img)
{
	cv::Mat gray;
	cv::cvtColor(img, gray, CV_BGR2GRAY);

	// Calculate gradients gx, gy
	cv::Mat gx, gy;
	Sobel(gray, gx, CV_32F, 1, 0, 1);
	Sobel(gray, gy, CV_32F, 0, 1, 1);

	cv::Mat gxThresh, gyThresh;

	cv::threshold(gx, gxThresh, -6.5, 128, CV_THRESH_BINARY_INV);
	cv::threshold(gy, gyThresh, -6.5, 127, CV_THRESH_BINARY_INV);

	// Compute gradient
	cv::Mat grad;
	cv::add(gxThresh, gyThresh, grad);
	cv::convertScaleAbs(grad, grad);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(grad, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// Get convex hull
	std::vector<std::vector<cv::Point> >hull(contours.size());

	// Verify hull contours
	std::vector<std::vector<cv::Point>> hullVerified;
	for (int i = 0; i < contours.size(); i++)
	{
		convexHull(cv::Mat(contours[i]), hull[i], false);
		if (cv::contourArea(hull[i]) > 50)
			hullVerified.push_back(hull[i]);
	}

	cv::Mat blank(img.size(), CV_8U, cv::Scalar(0));

	cv::drawContours(blank, hullVerified, -1, cv::Scalar(255), -1);

	// Erode blank
	cv::erode(blank, blank, cv::Mat(3, 3, CV_8U, cv::Scalar::all(1)), cv::Point(-1, -1));

	// Make mask bigger
	cv::dilate(blank, blank, cv::Mat(8, 8, CV_8U, cv::Scalar::all(1)), cv::Point(-1, -1), 5);

	// Find contours in blank
	std::vector<std::vector<cv::Point>> contoursBlank;
	cv::findContours(blank, contoursBlank, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// Get bounding rectangle
	std::vector<std::vector<cv::Point> > contours_poly(contoursBlank.size());
	std::vector<cv::Rect> boundRect(contoursBlank.size());

	// Image for drawing rectangles
	cv::Mat drawing(img.size(), CV_8U, cv::Scalar::all(0));

	// Get rectangles + find mean area of rectangle
	double meanArea = 0;
	for (int i = 0; i < contoursBlank.size(); i++)
	{
		approxPolyDP(cv::Mat(contoursBlank[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(cv::Mat(contours_poly[i]));
		meanArea += boundRect[i].width * boundRect[i].height;
	}
	meanArea /= contoursBlank.size();

	// Get rectangles which is bigger than mean area
	std::vector<cv::Mat> cashes;

	for (int i = 0; i < contoursBlank.size(); ++i)
	{
		if (boundRect[i].width * boundRect[i].height > meanArea)
		{
			cv::Mat temp = img(boundRect[i]);
			cashes.push_back(temp);
		}
	}

	return cashes;
}
/*
std::vector<size_t> detectTargetsAndOutputResults(cv::Mat& image, const std::string& imageFilename, bool useHighGUI) {
	cv::Mat imageBackup = image.clone();
	cv::Ptr< std::vector< cv::Ptr<DetectorResult> > > detectorResultsOut = detectTargets(image);
	std::vector<size_t> results;

	std::stringstream imageInliersOutputFilename;
	imageInliersOutputFilename << TEST_OUTPUT_DIRECTORY << imageFilename << FILENAME_SEPARATOR << _configurationTags << FILENAME_SEPARATOR << INLIERS_MATCHES << FILENAME_SEPARATOR;

	for (size_t i = 0; i < detectorResultsOut->size(); ++i) {
		Ptr<DetectorResult> detectorResult = (*detectorResultsOut)[i];
		results.push_back(detectorResult->getTargetValue());

		cv::drawKeypoints(image, detectorResult->getInliersKeypoints(), image, TARGET_KEYPOINT_COLOR);

		stringstream ss;
		ss << detectorResult->getTargetValue();

		Mat imageMatchesSingle = imageBackup.clone();
		Mat matchesInliers = detectorResult->getInliersMatches(imageMatchesSingle);

		try {
			Rect boundingBox = cv::boundingRect(detectorResult->getTargetContour());
			ImageUtils::correctBoundingBox(boundingBox, image.cols, image.rows);
			GUIUtils::drawLabelInCenterOfROI(ss.str(), image, boundingBox);
			GUIUtils::drawLabelInCenterOfROI(ss.str(), matchesInliers, boundingBox);
			ImageUtils::drawContour(image, detectorResult->getTargetContour(), detectorResult->getContourColor());
			ImageUtils::drawContour(matchesInliers, detectorResult->getTargetContour(), detectorResult->getContourColor());
		}
		catch (...) {
			std::cerr << "!!! Drawing outside image !!!" << endl;
		}

		if (useHighGUI) {
			stringstream windowName;
			windowName << "Target inliers matches (window " << i << ")";
			cv::namedWindow(windowName.str(), CV_WINDOW_KEEPRATIO);
			cv::imshow(windowName.str(), matchesInliers);
			cv::waitKey(10);
		}

		stringstream imageOutputFilenameFull;
		imageOutputFilenameFull << imageInliersOutputFilename.str() << i << IMAGE_OUTPUT_EXTENSION;
		imwrite(imageOutputFilenameFull.str(), matchesInliers);
	}

	sort(results.begin(), results.end());

	cout << "    -> Detected " << results.size() << (results.size() != 1 ? " targets" : " target");
	size_t globalResult = 0;
	stringstream resultsSS;
	if (!results.empty()) {
		resultsSS << " (";
		for (size_t i = 0; i < results.size(); ++i) {
			size_t resultValue = results[i];
			resultsSS << " " << resultValue;
			globalResult += resultValue;
		}
		resultsSS << " )";
		cout << resultsSS.str();
	}
	cout << endl;

	stringstream globalResultSS;
	globalResultSS << "Global result: " << globalResult << resultsSS.str();
	Rect globalResultBoundingBox(0, 0, image.cols, image.rows);
	GUIUtils::drawImageLabel(globalResultSS.str(), image, globalResultBoundingBox);

	stringstream imageOutputFilename;
	imageOutputFilename << TEST_OUTPUT_DIRECTORY << imageFilename << FILENAME_SEPARATOR << _configurationTags << IMAGE_OUTPUT_EXTENSION;
	imwrite(imageOutputFilename.str(), image);

	return results;
}
*/


int main(int argc, char** argv) {
	// cli
	CLI userInterface;
	userInterface.startInteractiveCLI();

	return 0;
}
