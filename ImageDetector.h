#pragma once
#define INLIERS_MATCHES "inliersMatches"
#define RESULTS_FILE "resultsAnalysis.txt"
#define RESULTS_FILE_HEADER ">>>>> Detector image results analysis <<<<<"
#define RESULTS_FILE_FOOTER ">>>>> Detector global results analysis <<<<<"
#define PRECISION_TOKEN "Precision"
#define RECALL_TOKEN "Recall"
#define ACCURACY_TOKEN "Accuracy"
#define GLOBAL_PRECISION_TOKEN "Global precision"
#define GLOBAL_RECALL_TOKEN "Global recall"
#define GLOBAL_ACCURACY_TOKEN "Global accuracy"
 
// std includes
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

// project includes
#include "Configs.h"
#include "ImagePreprocessor.h"
#include "DetectorEvaluationResult.h"
#include "GUI\GUIUtils.h"
#include "DetectorResult.h"
#include "libs/PerformanceTimer.h"
#include "TargetDetector.h"

// namespace specific imports to avoid namespace pollution
using std::cout;
using std::endl;
using std::string;
using std::stringstream;
using std::vector;
using std::ifstream;
using std::ofstream;

using cv::Mat;
using cv::Ptr;
using cv::Rect;
using cv::Vec2d;
using cv::FeatureDetector;
using cv::DescriptorExtractor;
using cv::DescriptorMatcher;
using cv::imwrite;

class ImageDetector {
public:
	ImageDetector(Ptr<FeatureDetector> _featureDetector, Ptr<DescriptorExtractor> _descriptorExtractor, Ptr<DescriptorMatcher> _descriptorMatcher, Ptr<ImagePreprocessor> _imagePreprocessor,
		const vector<string>& _referenceImagesDirectories,
		bool _useInliersGlobalMatch = true,
		const string& _referenceImagesListPath = REFERENCE_IMGAGES_LIST, const string& _testImagesListPath = TEST_IMGAGES_LIST);
	~ImageDetector();

	bool setupTargetDB(const string& _referenceImagesListPaths, bool _useInliersGlobalMatch = true);
	void setupTargetsShapesRanges(const string& _maskPath = TARGETS_SHAPE_MASKS);

	Ptr< vector< Ptr<DetectorResult> > > detectTargets(Mat& image, float minimumMatchAllowed = 0.07, float minimumTargetAreaPercentage = 0.05,
		float maxDistanceRatio = 0.75f, float reprojectionThresholdPercentage = 0.01f, double confidence = 0.999, int maxIters = 5000, size_t minimumNumberInliers = 8);
	vector<size_t> detectTargetsAndOutputResults(Mat& image, const string& imageFilename = "", bool useHighGUI = false);
	DetectorEvaluationResult evaluateDetector(const string& testImgsList, bool saveResults = true);

	void extractExpectedResultsFromFilename(string filename, vector<size_t>& expectedResultFromTestOut);

private:
	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorExtractor> descriptorExtractor;
	Ptr<DescriptorMatcher> descriptorMatcher;

	Ptr<ImagePreprocessor> imagePreprocessor;
	string configurationTags;
	vector<string> referenceImagesDirectories;
	string referenceImagesListPath;
	string testImagesListPath;

	vector<TargetDetector> targetDetectors;

	Vec2d contourAspectRatioRange;
	Vec2d contourCircularityRange;
};
