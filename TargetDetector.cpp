#include "TargetDetector.h"


TargetDetector::TargetDetector(Ptr<FeatureDetector> _featureDetector, Ptr<DescriptorExtractor> _descriptorExtractor, Ptr<DescriptorMatcher> _descriptorMatcher,
	size_t _targetTag, bool _useInliersGlobalMatch) :
	featureDetector(_featureDetector), descriptorExtractor(_descriptorExtractor), descriptorMatcher(_descriptorMatcher/*->clone(true)*/),
	targetTag(_targetTag), useInliersGlobalMatch(_useInliersGlobalMatch),
	currentLODIndex(0) {}

TargetDetector::~TargetDetector() {}


bool TargetDetector::setupTargetRecognition(const Mat& targetImage, const Mat& targetROIs) {
	targetsImage.push_back(targetImage);
	targetsKeypoints.push_back(vector<KeyPoint>());
	targetsDescriptors.push_back(Mat());
	currentLODIndex = targetsKeypoints.size() - 1;

	// detect target keypoints
	featureDetector->detect(targetsImage[currentLODIndex], targetsKeypoints[currentLODIndex], targetROIs);
	// as research shows 
	// if there in less than 4 keypoints 
	// it isn`t enough for detection
	if (targetsKeypoints[currentLODIndex].size() < 4) {
		return false;
	}

	// compute descriptors
	descriptorExtractor->compute(targetsImage[currentLODIndex], targetsKeypoints[currentLODIndex], targetsDescriptors[currentLODIndex]);
	if (targetsDescriptors[currentLODIndex].rows < 4) {
		return false; 
	}


	// train matcher to speedup recognition in case flann is used
	/*_descriptorMatcher->add(_targetDescriptors);
	_descriptorMatcher->train();*/

	// associate key points to ROIs
	if (!useInliersGlobalMatch) {
		return setupTargetROIs(targetsKeypoints[currentLODIndex], targetROIs);
	}
	else {
		return true;
	}
}


bool TargetDetector::setupTargetROIs(const vector<KeyPoint>& _targetKeypoints, const Mat& targetROIs) {
	targetKeypointsAssociatedROIsIndexes.push_back(vector<size_t>());
	numberOfKeypointInsideContours.push_back(vector<size_t>());

	if (_targetKeypoints.empty()) { return false; }

	targetKeypointsAssociatedROIsIndexes[currentLODIndex].resize(_targetKeypoints.size());

	vector< vector<Point> > targetROIsContours;
	vector<Vec4i> hierarchy;
	cv::findContours(targetROIs.clone(), targetROIsContours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	if (targetROIsContours.empty()) { return false; }

	int targetKeypointsSize = _targetKeypoints.size();
#pragma omp parallel for
	for (int targetKeypointsIndex = 0; targetKeypointsIndex < targetKeypointsSize; ++targetKeypointsIndex) {
		for (size_t contourIndex = 0; contourIndex < targetROIsContours.size(); ++contourIndex) {
			// point inside contour or in the border
			Point2f point = _targetKeypoints[targetKeypointsIndex].pt;
			if (cv::pointPolygonTest(targetROIsContours[contourIndex], point, false) >= 0) {
				targetKeypointsAssociatedROIsIndexes[currentLODIndex][targetKeypointsIndex] = contourIndex;
				break;
			}
		}
	}

	numberOfKeypointInsideContours[currentLODIndex].clear();
	numberOfKeypointInsideContours[currentLODIndex].resize(targetROIsContours.size(), 0);

	for (size_t i = 0; i < targetKeypointsAssociatedROIsIndexes[currentLODIndex].size(); ++i) {
		size_t contourIndex = targetKeypointsAssociatedROIsIndexes[currentLODIndex][i];
		++numberOfKeypointInsideContours[currentLODIndex][contourIndex];
	}

	return true;
}


void TargetDetector::updateCurrentLODIndex(const Mat& imageToAnalyze, float targetResolutionSelectionSplitOffset) {
	int halfImageResolution = imageToAnalyze.cols / 2;

	size_t newLODIndex = 0;
	for (size_t i = 1; i < targetsImage.size(); ++i) {
		int previousLODWidthResolution = targetsImage[i - 1].cols;
		int currentLODWidthResolution = targetsImage[i].cols;

		if (halfImageResolution > currentLODWidthResolution) {
			newLODIndex = i; // use bigger resolution
		}
		else if (halfImageResolution < previousLODWidthResolution) {
			newLODIndex = i - 1; // use lower resolution
			break;
		}
		else {
			int splittingPointResolutions = (int)((currentLODWidthResolution - previousLODWidthResolution) * targetResolutionSelectionSplitOffset);
			int imageOffsetResolution = currentLODWidthResolution - halfImageResolution;

			if (imageOffsetResolution < splittingPointResolutions) {
				newLODIndex = i - 1; // use lower resolution
				break;
			}
			else {
				newLODIndex = i; // use bigger resolution
				break;
			}
		}
	}

	currentLODIndex = newLODIndex;
}


Ptr<DetectorResult> TargetDetector::analyzeImage(const vector<KeyPoint>& keypointsQueryImage, const Mat& descriptorsQueryImage,
	float maxDistanceRatio, float reprojectionThreshold, double confidence, int maxIters, size_t minimumNumberInliers) {
	vector<DMatch> matches;
	ImageUtils::matchDescriptorsWithRatioTest(descriptorMatcher, descriptorsQueryImage, targetsDescriptors[currentLODIndex], matches, maxDistanceRatio);
	//_descriptorMatcher->match(descriptorsQueryImage, _targetDescriptors, matches);
	//_descriptorMatcher->match(descriptorsQueryImage, matches); // flann speedup

	if (matches.size() < minimumNumberInliers) {
		return new DetectorResult();
	}

	Mat homography;
	vector<DMatch> inliers;
	vector<unsigned char> inliersMaskOut;
	ImageUtils::refineMatchesWithHomography(keypointsQueryImage, targetsKeypoints[currentLODIndex], matches, homography, inliers, inliersMaskOut, reprojectionThreshold, confidence, maxIters, minimumNumberInliers);

	if (inliers.size() < minimumNumberInliers) {
		return new DetectorResult();
	}

	float bestROIMatch = 0;
	if (useInliersGlobalMatch) {
		bestROIMatch = (float)inliers.size() / (float)matches.size();
	}
	else {
		bestROIMatch = computeBestROIMatch(inliers, minimumNumberInliers);
	}

	return new DetectorResult(targetTag, vector<Point>(), bestROIMatch, targetsImage[currentLODIndex], targetsKeypoints[currentLODIndex], keypointsQueryImage, matches, inliers, inliersMaskOut, homography);
}


float TargetDetector::computeBestROIMatch(const vector<DMatch>& inliers, size_t minimumNumberInliers) {
	vector<size_t> roisInliersCounts(numberOfKeypointInsideContours[currentLODIndex].size(), 0);

	for (size_t i = 0; i < inliers.size(); ++i) {
		size_t roiIndex = targetKeypointsAssociatedROIsIndexes[currentLODIndex][inliers[i].trainIdx];
		++roisInliersCounts[roiIndex];
	}

	float bestROIMatch = 0;
	for (size_t i = 0; i < roisInliersCounts.size(); ++i) {
		size_t roiTotalCount = numberOfKeypointInsideContours[currentLODIndex][i];
		if (roiTotalCount != 0) {
			float roiMatch = (float)roisInliersCounts[i] / (float)roiTotalCount;
			if (roiMatch > bestROIMatch && roisInliersCounts[i] > minimumNumberInliers) {
				bestROIMatch = roiMatch;
			}
		}
	}

	return bestROIMatch;
}
