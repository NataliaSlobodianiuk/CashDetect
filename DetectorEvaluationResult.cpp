#include "DetectorEvaluationResult.h"


DetectorEvaluationResult::DetectorEvaluationResult() {}

DetectorEvaluationResult::DetectorEvaluationResult(double _precision, double _recall, double _accuracy) :
	precision(_precision), recall(_recall), accuracy(_accuracy) {}

DetectorEvaluationResult::DetectorEvaluationResult(size_t _truePositives, size_t _trueNegatives, size_t _falsePositives, size_t _falseNegatives) :
	truePositives(_truePositives), trueNegatives(_trueNegatives), falsePositives(_falsePositives), falseNegatives(_falseNegatives) {

	updateMeasures();
}

DetectorEvaluationResult::DetectorEvaluationResult(vector<size_t> results, vector<size_t> expectedResults) :
	truePositives(0), trueNegatives(0), falsePositives(0), falseNegatives(0) {
	std::sort(results.begin(), results.end());
	std::sort(expectedResults.begin(), expectedResults.end());

	for (size_t resultsIndex = 0; resultsIndex < results.size(); ++resultsIndex) {
		vector<size_t>::iterator it = std::find(expectedResults.begin(), expectedResults.end(), results[resultsIndex]);

		if (it != expectedResults.end()) {
			++truePositives;
			expectedResults.erase(it);
		}
		else {
			++falsePositives;
		}
	}

	falseNegatives = expectedResults.size();

	updateMeasures();
}

DetectorEvaluationResult::DetectorEvaluationResult(Mat& votingMask, vector<Mat>& targetMasks, unsigned short votingMaskThreshold) :
	truePositives(0), trueNegatives(0), falsePositives(0), falseNegatives(0) {
	Mat mergedTargetsMask;
	if (ImageUtils::mergeTargetMasks(targetMasks, mergedTargetsMask)) {
		computeMasksSimilarity(votingMask, mergedTargetsMask, votingMaskThreshold, &truePositives, &trueNegatives, &falsePositives, &falseNegatives);

		updateMeasures();
	}
}


bool DetectorEvaluationResult::computeMasksSimilarity(Mat& votingMask, Mat& mergedTargetsMask, unsigned short votingMaskThreshold,
	size_t* truePositivesOut, size_t* trueNegativesOut, size_t* falsePositivesOut, size_t* falseNegativesOut) {
	if (votingMask.rows == mergedTargetsMask.rows && votingMask.cols == mergedTargetsMask.cols) {
		size_t truePositives = 0;
		size_t trueNegatives = 0;
		size_t falsePositives = 0;
		size_t falseNegatives = 0;

#pragma omp parallel for schedule(dynamic)
		for (int votingMaskY = 0; votingMaskY < votingMask.rows; ++votingMaskY) {
			for (int votingMaskX = 0; votingMaskX < votingMask.cols; ++votingMaskX) {
				if (votingMask.at<unsigned short>(votingMaskY, votingMaskX) > votingMaskThreshold) {
					if (mergedTargetsMask.at<unsigned char>(votingMaskY, votingMaskX) > 0) {
#pragma omp atomic
						++truePositives;
					}
					else {
#pragma omp atomic
						++falsePositives;
					}
				}
				else {
					if (mergedTargetsMask.at<unsigned char>(votingMaskY, votingMaskX) > 0) {
#pragma omp atomic
						++falseNegatives;
					}
					else {
#pragma omp atomic
						++trueNegatives;
					}
				}
			}
		}

		*truePositivesOut = truePositives;
		*trueNegativesOut = trueNegatives;
		*falsePositivesOut = falsePositives;
		*falseNegativesOut = falseNegatives;
		return true;
	}

	return false;
}


double DetectorEvaluationResult::computePrecision(size_t truePositives, size_t falsePositives) {
	double divisor = truePositives + falsePositives;
	if (divisor == 0) {
		return 0;
	}

	return (double)truePositives / divisor;
}


double DetectorEvaluationResult::computeRecall(size_t truePositives, size_t falseNegatives) {
	double divisor = truePositives + falseNegatives;
	if (divisor == 0) {
		return 0;
	}

	return (double)truePositives / divisor;
}


double DetectorEvaluationResult::computeAccuracy(size_t truePositives, size_t trueNegatives, size_t falsePositives, size_t falseNegatives) {
	double divisor = truePositives + trueNegatives + falsePositives + falseNegatives;
	if (divisor == 0) {
		return 0;
	}

	return (double)(truePositives + trueNegatives) / divisor;
}


void DetectorEvaluationResult::updateMeasures() {
	precision = computePrecision(truePositives, falsePositives);
	recall = computeRecall(truePositives, falseNegatives);
	accuracy = computeAccuracy(truePositives, trueNegatives, falsePositives, falseNegatives);
}

