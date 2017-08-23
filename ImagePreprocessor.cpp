#include "ImagePreprocessor.h"


ImagePreprocessor::ImagePreprocessor(int _claehClipLimit, int _claehTileXSize, int _claehTileYSize,
	int _bilateralFilterDistance, int _bilateralFilterSigmaColor, int _bilateralFilterSigmaSpace,
	int _contrastMultipliedBy10, int _brightnessMultipliedBy10, int _GAUSSIAN_SMOOTH_FILTER_SIZE,
	int _ADAPTIVE_THRESH_BLOCK_SIZE, int _ADAPTIVE_THRESH_WEIGHT) :
	claehClipLimit(_claehClipLimit), claehTileXSize(_claehTileXSize), claehTileYSize(_claehTileYSize),
	bilateralFilterDistance(_bilateralFilterDistance), bilateralFilterSigmaColor(_bilateralFilterSigmaColor), bilateralFilterSigmaSpace(_bilateralFilterSigmaSpace),
	contrast(_contrastMultipliedBy10), brightness(_brightnessMultipliedBy10), GAUSSIAN_SMOOTH_FILTER_SIZE(_GAUSSIAN_SMOOTH_FILTER_SIZE),
	ADAPTIVE_THRESH_BLOCK_SIZE(_ADAPTIVE_THRESH_BLOCK_SIZE), ADAPTIVE_THRESH_WEIGHT(_ADAPTIVE_THRESH_WEIGHT){}

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


cv::Mat ImagePreprocessor::extractValue(cv::Mat &imgOriginal) {
	cv::Mat imgHSV;
	std::vector<cv::Mat> vectorOfHSVImages;
	cv::Mat imgValue;

	cv::cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);

	cv::split(imgHSV, vectorOfHSVImages);

	imgValue = vectorOfHSVImages[2];

	return(imgValue);
}


cv::Mat ImagePreprocessor::maximizeContrast(cv::Mat &imgGrayscale) {
	cv::Mat imgTopHat;
	cv::Mat imgBlackHat;
	cv::Mat imgGrayscalePlusTopHat;
	cv::Mat imgGrayscalePlusTopHatMinusBlackHat;

	cv::Mat structuringElement = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3));

	cv::morphologyEx(imgGrayscale, imgTopHat, CV_MOP_TOPHAT, structuringElement);
	cv::morphologyEx(imgGrayscale, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);

	imgGrayscalePlusTopHat = imgGrayscale + imgTopHat;
	imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - imgBlackHat;

	return(imgGrayscalePlusTopHatMinusBlackHat);
}


void ImagePreprocessor::preprocessImage(Mat& image, bool useCVHighGUI)
{
	if (image.channels() == 3) {
		// extract value channel only from original image to get imgGrayscale
		cv::Mat gray = extractValue(image);

		// maximize contrast with top hat and black hat
		cv::Mat imgMaxContrastGrayscale = maximizeContrast(gray);

		cv::Mat imgBlurred;

		// Create size for Gaussian blur
		cv::Size GAUSSIAN(GAUSSIAN_SMOOTH_FILTER_SIZE, GAUSSIAN_SMOOTH_FILTER_SIZE);

		// gaussian blur
		cv::GaussianBlur(imgMaxContrastGrayscale, imgBlurred, GAUSSIAN, 0);

		// call adaptive threshold to get imgThresh
		cv::Mat thresh;
		cv::adaptiveThreshold(imgBlurred, thresh, 255.0, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE * 2 + 1, ADAPTIVE_THRESH_WEIGHT);

		cv::dilate(thresh, thresh, cv::Mat(35, 35, CV_8U, cv::Scalar::all(1)));

		cv::Mat blank(image.size(), image.type(), cv::Scalar::all(0));

		// Mask image - get parts in which we are interested in 
		image.copyTo(blank, thresh);

		image = blank.clone();
	}

	// remove noise with bilateral filter
	cv::bilateralFilter(image.clone(), image, bilateralFilterDistance, bilateralFilterSigmaColor, bilateralFilterSigmaSpace);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_BILATERAL_FILTER, image);
	}

	// histogram equalization to improve color segmentation
	//histogramEqualization(image.clone(), false, useCVHighGUI);
	histogramEqualization(image, true, useCVHighGUI);

	// increase contrast and brightness
	image.convertTo(image, -1, (double)contrast / 10.0, (double)brightness / 10.0);

	cv::bilateralFilter(image.clone(), image, bilateralFilterDistance, bilateralFilterSigmaColor, bilateralFilterSigmaSpace);
	if (useCVHighGUI) {
		//imshow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS, image);
	}
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

	
	if (useCVHighGUI) {
		if (useCLAHE) {
			imshow(WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE, image);
		}
		else {
			imshow(WINDOW_NAME_HISTOGRAM_EQUALIZATION, image);
		}
	}
}