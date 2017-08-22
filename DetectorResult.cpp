#include "DetectorResult.h"

DetectorResult::DetectorResult() : bestROIMatch(0) {}

DetectorResult::DetectorResult(size_t _targetValue, const vector<Point>& _targetContour, float _bestROIMatch,
	const Mat& _referenceImage, const vector<KeyPoint>& _referenceImageKeypoints, const vector<KeyPoint>& _keypointsQueryImage,
	const vector<DMatch>& _matches, const vector<DMatch>& _inliers, const vector<unsigned char>& _inliersMatchesMask, const Mat& _homography) :

	targetValue(_targetValue), targetContour(_targetContour), bestROIMatch(_bestROIMatch),
	referenceImage(_referenceImage), referenceImageKeypoints(_referenceImageKeypoints), keypointsQueryImage(_keypointsQueryImage),
	matches(_matches), inliers(_inliers), inliersMatchesMask(_inliersMatchesMask), homography(_homography) {}


DetectorResult::~DetectorResult() {}


vector<Point>& DetectorResult::getTargetContour() {
	if (targetContour.empty()) {
		vector<Point2f> corners;
		corners.push_back(Point2f(0.0f, 0.0f));
		corners.push_back(Point2f((float)referenceImage.cols, 0.0f));
		corners.push_back(Point2f((float)referenceImage.cols, (float)referenceImage.rows));
		corners.push_back(Point2f(0.0f, (float)referenceImage.rows));

		vector<Point2f> transformedCorners;
		cv::perspectiveTransform(corners, transformedCorners, homography);

		for (size_t i = 0; i < transformedCorners.size(); ++i) {
			targetContour.push_back(Point((int)transformedCorners[i].x, (int)transformedCorners[i].y));
		}
	}

	return targetContour;
}


vector<KeyPoint>& DetectorResult::getInliersKeypoints() {
	if (inliersKeyPoints.empty()) {
		for (size_t i = 0; i < inliers.size(); ++i) {
			DMatch match = inliers[i];

			if ((size_t)match.queryIdx < keypointsQueryImage.size()) {
				inliersKeyPoints.push_back(keypointsQueryImage[match.queryIdx]);
			}
		}
	}

	return inliersKeyPoints;
}


Mat DetectorResult::getInliersMatches(Mat& queryImage) {
	Mat inliersMatches;

	if (inliers.empty()) {
		return queryImage;
	}
	else {
		cv::drawMatches(queryImage, keypointsQueryImage, referenceImage, referenceImageKeypoints, inliers, inliersMatches, TARGET_KEYPOINT_COLOR, NONTARGET_KEYPOINT_COLOR);
		return inliersMatches;
	}
}

