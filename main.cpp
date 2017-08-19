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

#include"ImagePreprocessor.h"

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



int main()
{	
	/*// Cash segmentation
	// Works good with simple cash
	// Work bad with rotated cash
	// C++ gradient calculation. 
	// Read image
	cv::Mat img = cv::imread("cash4.jpg");

	int sum = 0;

	// Filenames of templates to be looking for
	std::vector<std::string> filenames = { "1_small.bmp", "2_small.bmp","5_small.bmp", 
	"10_small.bmp", "20_small.bmp", "50_small.bmp", "100_small.bmp", "200_small.bmp", "500_small.bmp" };

	// Get cash extracted
	std::vector<cv::Mat> cash = extractCash(img);

	std::cout << cash.size() << std::endl;

	// Create detector and extractor
	int minHessian = 400;
	cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> detector = cv::xfeatures2d::SurfFeatureDetector::create(minHessian);
	cv::Ptr<cv::xfeatures2d::SurfDescriptorExtractor> extractor = cv::xfeatures2d::SurfDescriptorExtractor::create();
	for (int c = 0; c < cash.size(); ++c) {
		// Get scene image
		cv::Mat img_scene;
		cv::cvtColor(cash[c], img_scene, CV_BGR2GRAY);
		
		// Detect keypoint in scene  
		std::vector<cv::KeyPoint> keypoints_scene;
		detector->detect(img_scene, keypoints_scene);

		// Compute
		cv::Mat descriptors_scene;
		extractor->compute(img_scene, keypoints_scene, descriptors_scene);

		// Perform mathing with templates
		for (int filename = 0; filename < filenames.size(); ++filename) {
			cv::Mat img_object = cv::imread(filenames[filename], 0);

			std::cout << filenames[filename] << std::endl;

			//-- Step 1: Detect the keypoints using SURF Detector
			
			std::vector<cv::KeyPoint> keypoints_object;
			detector->detect(img_object, keypoints_object);

			//-- Step 2: Calculate descriptors (feature vectors)
			cv::Mat descriptors_object;

			extractor->compute(img_object, keypoints_object, descriptors_object);
		
			//-- Step 3: Matching descriptor vectors using FLANN matcher
			cv::FlannBasedMatcher matcher;
			std::vector< cv::DMatch > matches;
			matcher.match(descriptors_object, descriptors_scene, matches);

			double max_dist = 0; double min_dist = 100;

			//-- Quick calculation of max and min distances between keypoints
			for (int i = 0; i < descriptors_object.rows; i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}

			printf("-- Max dist : %f \n", max_dist);
			printf("-- Min dist : %f \n", min_dist);

			//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
			std::vector< cv::DMatch > good_matches;

			for (int i = 0; i < descriptors_object.rows; i++)
			{
				if (matches[i].distance < 3 * min_dist)
				{
					good_matches.push_back(matches[i]);
				}
			}

			// Check if there is good matches
			if (!good_matches.empty()) {
				cv::Mat img_matches;
				drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches,
					cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

				//-- Localize the object
				std::vector<cv::Point2f> obj;
				std::vector<cv::Point2f> scene;

				for (int i = 0; i < good_matches.size(); i++)
				{
					//-- Get the keypoints from the good matches
					obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
					scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
				}

				cv::Mat H = findHomography(obj, scene, cv::RANSAC);

				//-- Get the corners from the image_1 ( the object to be "detected" )
				std::vector<cv::Point2f> obj_corners(4);
				obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
				obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
				std::vector<cv::Point2f> scene_corners(4);

				if (!H.empty()) {
					perspectiveTransform(obj_corners, scene_corners, H);
					
					// Verify lines
					if (verifyRectangle(img, scene_corners[0], scene_corners[1], scene_corners[2], scene_corners[3])) {
						//-- Draw lines between the corners (the mapped object in the scene - image_2 )
						line(img_matches, scene_corners[0] + cv::Point2f(img_object.cols, 0), scene_corners[1] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
						line(img_matches, scene_corners[1] + cv::Point2f(img_object.cols, 0), scene_corners[2] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
						line(img_matches, scene_corners[2] + cv::Point2f(img_object.cols, 0), scene_corners[3] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
						line(img_matches, scene_corners[3] + cv::Point2f(img_object.cols, 0), scene_corners[0] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);

						if (filename == 0)
						{
							std::cout << 1 << std::endl;
							sum += 1;
						}
						if (filename == 1)
						{
							std::cout << 2 << std::endl;
							sum += 2;
						}
						if (filename == 2)
						{
							std::cout << 5 << std::endl;
							sum += 5;
						}
						if (filename == 3)
						{
							std::cout << 10 << std::endl;
							sum += 10;
						}
						if (filename == 4)
						{
							std::cout << 20 << std::endl;
							sum += 20;
						}
						if (filename == 5)
						{
							std::cout << 50 << std::endl;
							sum += 50;
						}
						if (filename == 6)
						{
							std::cout << 100 << std::endl;
							sum += 100;
						}
						if (filename == 7)
						{
							std::cout << 200 << std::endl;
							sum += 200;
						}
						if (filename == 8)
						{
							std::cout << 500 << std::endl;
							sum += 500;
						}
						//-- Show detected matches
						cv::namedWindow("Good Matches & Object detection", CV_WINDOW_FREERATIO);
						imshow("Good Matches & Object detection", img_matches);
					}
				}
				else {
					std::cout << "Homography matrix is empty" << std::endl;
				}
			}
			else {
				std::cout << "There is no good matches" << std::endl;
			}

			cv::waitKey(0);
		}
	}

	std::cout << "Sum " << sum << std::endl;
	*/

	cv::Mat image = cv::imread("cash15.jpg");

	ImagePreprocessor preprocessor;

	preprocessor.preprocessImage(image);

	cv::namedWindow("Preprocessed image", CV_WINDOW_FREERATIO);
	cv::imshow("Preprocessed image", image);
	cv::waitKey(0);
	return 0;
}