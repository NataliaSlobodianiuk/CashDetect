#pragma once
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>

// project includes
#include "ImageUtils.h"


// namespace specific imports to avoid namespace pollution
using std::vector;

using cv::Mat;
using cv::Vec3b;


class DetectorEvaluationResult {
public:
	DetectorEvaluationResult();
	DetectorEvaluationResult(double _precision, double _recall, double _accuracy);
	DetectorEvaluationResult(size_t _truePositives, size_t _trueNegatives, size_t _falsePositives, size_t _falseNegatives);
	DetectorEvaluationResult(vector<size_t> results, vector<size_t> expectedResults);
	DetectorEvaluationResult(Mat& _votingMask, vector<Mat>& _targetMasks, unsigned short _votingMaskThreshold = 1);
	~DetectorEvaluationResult() {}

	static bool computeMasksSimilarity(Mat& votingMask, Mat& mergedTargetsMask, unsigned short votingMaskThreshold,
		size_t* truePositivesOut, size_t* trueNegativesOut, size_t* falsePositivesOut, size_t* falseNegativesOut);

	static double computePrecision(size_t truePositives, size_t falsePositives);
	static double computeRecall(size_t truePositives, size_t falseNegatives);
	static double computeAccuracy(size_t truePositives, size_t trueNegatives, size_t falsePositives, size_t falseNegatives);

	void updateMeasures();

	double getPrecision() const { return precision; }
	void setPrecision(double val) { precision = val; }
	double getRecall() const { return recall; }
	void setRecall(double val) { recall = val; }
	double getAccuracy() const { return accuracy; }
	void setAccuracy(double val) { accuracy = val; }

	size_t getTruePositives() const { return truePositives; }
	void setTruePositives(size_t val) { truePositives = val; }
	size_t getTrueNegatives() const { return trueNegatives; }
	void setTrueNegatives(size_t val) { trueNegatives = val; }
	size_t getFalsePositives() const { return falsePositives; }
	void setFalsePositives(size_t val) { falsePositives = val; }
	size_t getFalseNegatives() const { return falseNegatives; }
	void setFalseNegatives(size_t val) { falseNegatives = val; }

private:
	double precision;	// truePositives / (truePositives + falsePositives)
	double recall;		// truePositives / (truePositives + falseNegatives)
	double accuracy;	// (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)

	size_t truePositives;
	size_t trueNegatives;
	size_t falsePositives;
	size_t falseNegatives;
};

