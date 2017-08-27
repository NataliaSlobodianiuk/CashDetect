#pragma once
// This implementation resulted from the refactoring of code distributed in the OpenCV 2.4.8
// The refactoring was performed to allow the fine tunning of the findHomography function


// std includes
#include <opencv2/core.hpp>

// project includes
#include "ModelEstimator.h"


class HomographyEstimator : public ModelEstimator {
	public:
		HomographyEstimator(int modelPoints);
		virtual ~HomographyEstimator();

		virtual int runKernel(const CvMat* m1, const CvMat* m2, CvMat* model);
		virtual bool refine(const CvMat* m1, const CvMat* m2, CvMat* model, int maxIters);


	protected:
		virtual void computeReprojError(const CvMat* m1, const CvMat* m2, const CvMat* model, CvMat* error);
};
