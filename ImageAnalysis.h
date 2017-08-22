#pragma once
// std includes
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

// project includes
#include "ImagePreprocessor.h"
#include "ImageDetector.h"
#include "GUI/GUIUtils.h"
#include "CLI.h"
// namespace specific imports to avoid namespace pollution
using std::string;
using std::stringstream;
using std::vector;
using std::map;
using std::pair;
using std::cout;

using cv::Mat;
using cv::Rect;
using cv::RotatedRect;
using cv::Ptr;
using cv::Scalar;
using cv::Vec3f;
using cv::Point;
using cv::Point2f;
using cv::Size;
using cv::VideoCapture;
using cv::imread;
using cv::waitKey;
using cv::imshow;
using cv::namedWindow;
using cv::moveWindow;
using cv::resizeWindow;
using cv::circle;
using cv::ellipse;
using cv::rectangle;


// Image analysis class that detects speed limits signs and recognizes the speed limit number
class ImageAnalysis {
public:

	// Constructor with initialization of parameters with default value		 		 
	ImageAnalysis(Ptr<ImagePreprocessor> imagePreprocessor, Ptr<ImageDetector> imageDetector);

	// ImageAnalysis destructor that performs cleanup of OpenCV HighGUI windows (in case they are used)		 
	~ImageAnalysis();


	/*!
	* \brief Processes the image from the specified path
	* \param filename Image name
	* \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
	* \return true if image was successfully processed
	*/
	bool processImage(string filename, bool useCVHighGUI = false);


	/*!
	* \brief Processes the image already loaded
	* \param image Image loaded and ready to be processed
	* \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
	* \return true if image was successfully processed
	*/
	bool processImage(Mat& image, bool useCVHighGUI = false);


	/*!
	* \brief Processes the image to reflect any internal parameter change
	* \return True if processing finished successfully
	*/
	bool updateImage();


	// brief Setups the HighGUI window were the original image is going to be drawn		 		 
	void setupMainWindow();


	/*!
	* \brief Setups the windows were the results will be presented
	* \param optionsOneWindow Flag to indicate to group the track bars in one window
	*/


	void setupResultsWindows(bool optionsOneWindow = false);
	void setScreenWidth(int val) { screenWidth = val; }
	void setScreenHeight(int val) { screenHeight = val; }
	void setOptionsOneWindow(bool val) { optionsOneWindow = val; }

private:
	Mat originalImage;
	Mat preprocessedImage;
	Mat processedImage;
	bool useCVHiGUI;
	bool windowsInitialized;
	bool optionsOneWindow;

	int screenWidth;
	int screenHeight;

	Ptr<ImagePreprocessor> imagePreprocessor;
	Ptr<ImageDetector> imageDetector;

	string filename;
};

