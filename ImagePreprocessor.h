#pragma once
#include <vector>
#include <string>
#include <iostream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

// project includes
#include "Configs.h"

// namespace specific imports to avoid namespace pollution
using std::vector;
using std::string;

using cv::Mat;
using cv::Size;
using cv::imshow;


class ImagePreprocessor {
public:
	ImagePreprocessor(int _claehClipLimit = 2, int _claehTileXSize = 4, int _claehTileYSize = 4,
		int _bilateralFilterDistance = 8, int _bilateralFilterSigmaColor = 16, int _bilateralFilterSigmaSpace = 12,
		int _contrastMultipliedBy10 = 9, int _brightnessMultipliedBy10 = 24);
	~ImagePreprocessor();


	bool loadAndPreprocessImage(const string& filename, Mat& imageLoadedOut, int loadFlags = CV_LOAD_IMAGE_COLOR, bool useCVHighGUI = false);

	/*!
	* \brief Preprocesses the image by applying bilateral filtering, histogram equalization, contrast and brightness correction and bilateral filtering again
	* \param image Image to be preprocessed
	* \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
	*/
	void preprocessImage(Mat& image, bool useCVHighGUI = true);


	/*!
	* \brief Applies histogram equalization to the specified image
	* \param image Image to equalize
	* \param useCLAHE If true, uses the contrast limited adaptive histogram equalization (CLAHE)
	* \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
	* \return
	*/
	void histogramEqualization(Mat& image, bool useCLAHE = true, bool useCVHighGUI = true);

	// Getters and setters
	int getClaehClipLimit() const { return claehClipLimit; }
	int* getClaehClipLimitPtr() { return &claehClipLimit; }
	void setClaehClipLimit(int val) { claehClipLimit = val; }
	int getClaehTileXSize() const { return claehTileXSize; }
	int* getClaehTileXSizePtr() { return &claehTileXSize; }
	void ClaehTileXSize(int val) { claehTileXSize = val; }
	int getClaehTileYSize() const { return claehTileYSize; }
	int* getClaehTileYSizePtr() { return &claehTileYSize; }
	void setClaehTileYSize(int val) { claehTileYSize = val; }

	int getBilateralFilterDistance() const { return bilateralFilterDistance; }
	int* getBilateralFilterDistancePtr() { return &bilateralFilterDistance; }
	void setBilateralFilterDistance(int val) { bilateralFilterDistance = val; }
	int getBilateralFilterSigmaColor() const { return bilateralFilterSigmaColor; }
	int* getBilateralFilterSigmaColorPtr() { return &bilateralFilterSigmaColor; }
	void setBilateralFilterSigmaColor(int val) { bilateralFilterSigmaColor = val; }
	int getBilateralFilterSigmaSpace() const { return bilateralFilterSigmaSpace; }
	int* getBilateralFilterSigmaSpacePtr() { return &bilateralFilterSigmaSpace; }
	void setBilateralFilterSigmaSpace(int val) { bilateralFilterSigmaSpace = val; }

	int getContrast() const { return contrast; }
	int* getContrastPtr() { return &contrast; }
	void setContrast(int val) { contrast = val;
	}
	int getBrightness() const { return brightness; }
	int* getBrightnessPtr() { return &brightness; }
	void setBrightness(int val) { brightness = val; }
	
private:
	int claehClipLimit;
	int claehTileXSize;
	int claehTileYSize;

	int bilateralFilterDistance;
	int bilateralFilterSigmaColor;
	int bilateralFilterSigmaSpace;

	int contrast;
	int brightness;
};

