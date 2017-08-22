#pragma once
// std includes
#include <iostream>
#include <string>
#include <sstream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

// project includes
#include "Configs.h"
#include "ConsoleInput.h"
#include "ImageAnalysis.h"
#include "ImageDetector.h"


// namespace specific imports to avoid namespace pollution
using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::stringstream;

using cv::FeatureDetector;
using cv::DescriptorExtractor;
using cv::DescriptorMatcher;
using cv::BOWTrainer;


// Command line interface
class CLI {
public:
	CLI() : imagePreprocessor(new ImagePreprocessor()) {}
	virtual ~CLI() {}

	void startInteractiveCLI();
	void showConsoleHeader();
	int getUserOption();
	void setupImageRecognition();

	int selectImagesDBLevelOfDetail();
	int selectInliersSelectionMethod();
	int selectFeatureDetector();
	int selectDescriptorExtractor();
	int selectDescriptorMatcher();

	void showVersion();

private:
	Ptr<ImagePreprocessor> imagePreprocessor;
	Ptr<ImageDetector> imageDetector;
};

