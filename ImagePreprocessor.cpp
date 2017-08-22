#include "ImagePreprocessor.h"


ImagePreprocessor::ImagePreprocessor(int _claehClipLimit, int _claehTileXSize, int _claehTileYSize,
	int _bilateralFilterDistance, int _bilateralFilterSigmaColor, int _bilateralFilterSigmaSpace,
	int _contrastMultipliedBy10, int _brightnessMultipliedBy10) :
	claehClipLimit(_claehClipLimit), claehTileXSize(_claehTileXSize), claehTileYSize(_claehTileYSize),
	bilateralFilterDistance(_bilateralFilterDistance), bilateralFilterSigmaColor(_bilateralFilterSigmaColor), bilateralFilterSigmaSpace(_bilateralFilterSigmaSpace),
	contrast(_contrastMultipliedBy10), brightness(_brightnessMultipliedBy10)
{}

ImagePreprocessor::~ImagePreprocessor() {}


bool ImagePreprocessor::loadAndPreprocessImage(const string& filename, Mat& imageLoadedOut, int loadFlags, bool useCVHighGUI) {
	if (filename != "") {
		try {
			//std::cout << filename << std::endl;
			imageLoadedOut = cv::imread(filename, loadFlags);
			if (!imageLoadedOut.data) { return false; }
			preprocessImage(imageLoadedOut, useCVHighGUI);
			return true;
		}
		catch (...) {
			return false;
		}
	}

	return false;
}


void ImagePreprocessor::preprocessImage(Mat& image, bool useCVHighGUI) {
	// remove noise with bilateral filter
	cv::bilateralFilter(image.clone(), image, bilateralFilterDistance, bilateralFilterSigmaColor, bilateralFilterSigmaSpace);
	/*if (useCVHighGUI) {
		imshow(WINDOW_NAME_BILATERAL_FILTER, image);
	}*/

	// histogram equalization to improve color segmentation
	//histogramEqualization(image.clone(), false, useCVHighGUI);
	histogramEqualization(image, true, useCVHighGUI);

	// increase contrast and brightness
	image.convertTo(image, -1, (double)contrast / 10.0, (double)brightness / 10.0);

	cv::bilateralFilter(image.clone(), image, bilateralFilterDistance, bilateralFilterSigmaColor, bilateralFilterSigmaSpace);
	/*if (useCVHighGUI) {
		//imshow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS, image);
	}*/
}


// apply histogramEqualization for better color segmentation using CLAHE or equalizeHist
void ImagePreprocessor::histogramEqualization(Mat& image, bool useCLAHE, bool useCVHighGUI) {
	vector<Mat> channels;
	if (image.channels() > 1) {
		cvtColor(image, image, CV_BGR2YCrCb);
		cv::split(image, channels);
	}

	if (useCLAHE) {
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE((claehClipLimit < 1 ? 1 : claehClipLimit), Size((claehTileXSize < 1 ? 1 : claehTileXSize), (claehTileYSize < 1 ? 1 : claehTileYSize)));
		if (image.channels() > 1) {
			clahe->apply(channels[0], channels[0]);
		}
		else {
			clahe->apply(image, image);
		}
	}
	else {
		cv::equalizeHist(channels[0], channels[0]);
	}

	if (image.channels() > 1) {
		cv::merge(channels, image);
		cvtColor(image, image, CV_YCrCb2BGR);
	}

	/*
	if (useCVHighGUI) {
		if (useCLAHE) {
			imshow(WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE, image);
		}
		else {
			imshow(WINDOW_NAME_HISTOGRAM_EQUALIZATION, image);
		}
	}*/
}