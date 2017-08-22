#pragma once
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

// project includes
#include "Configs.h"

// namespace specific imports to avoid namespace pollution
using std::vector;

using cv::Mat;
using cv::Point;
using cv::Point2f;
using cv::Scalar;
using cv::DMatch;
using cv::KeyPoint;

class DetectorResult {
public:
	DetectorResult();
	DetectorResult(size_t _targetValue, const vector<Point>& _targetContour, float _bestROIMatch,
		const Mat& _referenceImage, const vector<KeyPoint>& _referenceImageKeypoints, const vector<KeyPoint>& _keypointsQueryImage,
		const vector<DMatch>& _matches, const vector<DMatch>& _inliers, const vector<unsigned char>& _inliersMatchesMask, const Mat& _homography);

	~DetectorResult();

	size_t& getTargetValue() { return targetValue; }
	vector<Point>& getTargetContour();
	float& getBestROIMatch() { return bestROIMatch; }
	Mat& getReferenceImage() { return referenceImage; }
	vector<KeyPoint>& getKeypointsQueryImage() { return keypointsQueryImage; }
	vector<DMatch>& getMatches() { return matches; }
	vector<DMatch>& getInliers() { return inliers; }
	vector<KeyPoint>& getInliersKeypoints();
	Mat getInliersMatches(Mat& queryImage);
	vector<unsigned char>& getInliersMatchesMask() { return inliersMatchesMask; }
	Mat& getHomography() { return homography; }

private:
	size_t targetValue;
	vector<Point> targetContour;
	float bestROIMatch;

	Mat referenceImage;
	vector<KeyPoint> referenceImageKeypoints;
	vector<KeyPoint> keypointsQueryImage;
	vector<DMatch> matches;
	vector<DMatch> inliers;
	vector<KeyPoint> inliersKeyPoints;
	vector<unsigned char> inliersMatchesMask;

	Mat homography;
};


