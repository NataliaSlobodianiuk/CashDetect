#include "ImageAnalysis.h"


ImageAnalysis::ImageAnalysis(Ptr<ImagePreprocessor> _imagePreprocessor, Ptr<ImageDetector> _imageDetector) :
	useCVHiGUI(true), windowsInitialized(false), optionsOneWindow(false), screenWidth(1920), screenHeight(1080),
	imagePreprocessor(_imagePreprocessor), imageDetector(_imageDetector) {};


// Destructor destroys all winndows in case 
// Opencv HighGUI was used
ImageAnalysis::~ImageAnalysis() {
	if (useCVHiGUI) {
		cv::destroyAllWindows();
	}
}


bool ImageAnalysis::processImage(string _filename, bool _useCVHighGUI) {
	Mat imageToProcess;
	bool loadSuccessful = true;
	// try to load image 
	if (_filename != "") {
		try {
			imageToProcess = imread(TEST_IMGAGES_DIRECTORY + _filename, CV_LOAD_IMAGE_COLOR);
		}
		catch (...) {
			loadSuccessful = false;
		}

		if (!imageToProcess.data) {
			loadSuccessful = false;
		}
	}
	else {
		loadSuccessful = false;
	}

	if (!loadSuccessful) {
		if (_useCVHighGUI) {
			cv::destroyAllWindows();
		}

		return false;
	}

	useCVHiGUI = _useCVHighGUI;
	windowsInitialized = false;

	// preprocss image
	filename = _filename;
	bool status = processImage(imageToProcess, _useCVHighGUI);
	filename = "";

	while (waitKey(10) != ESC_KEYCODE) {}

	if (_useCVHighGUI) {
		cv::destroyAllWindows();
	}

	return status;
}


bool ImageAnalysis::processImage(Mat& image, bool _useCVHighGUI) {
	originalImage = image.clone();
	useCVHiGUI = _useCVHighGUI;

	// show original image in case  _useCVHighGUI == true
	if (_useCVHighGUI) {
		if (!windowsInitialized) {
			setupMainWindow();
			setupResultsWindows(optionsOneWindow);
			windowsInitialized = true;
		}

		imshow(WINDOW_NAME_MAIN, originalImage);
	}

	preprocessedImage = image.clone();
	imagePreprocessor->preprocessImage(preprocessedImage, _useCVHighGUI);
	processedImage = preprocessedImage.clone();

	imageDetector->detectTargetsAndOutputResults(processedImage, filename, true);

	// show processed image in case _useCVHighGUI == true
	if (_useCVHighGUI) {
		imshow(WINDOW_NAME_TARGET_DETECTION, processedImage);
	}

	return true;
}


bool ImageAnalysis::updateImage() {
	return processImage(originalImage.clone(), useCVHiGUI);
}

void updateImageAnalysis(int position, void* userData) {
	ImageAnalysis* imgAnalysis = ((ImageAnalysis*)userData);
	imgAnalysis->updateImage();
}


void ImageAnalysis::setupMainWindow() {
	GUIUtils::addHighGUIWindow(0, 0, WINDOW_NAME_MAIN, originalImage.size().width, originalImage.size().height, screenWidth, screenHeight);
}


void ImageAnalysis::setupResultsWindows(bool optionsOneWindow) {
	GUIUtils::addHighGUIWindow(1, 0, WINDOW_NAME_BILATERAL_FILTER, originalImage.size().width, originalImage.size().height, screenWidth, screenHeight);
	//GUIUtils::addHighGUIWindow(2, 0, WINDOW_NAME_HISTOGRAM_EQUALIZATION, _originalImage.size().width, _originalImage.size().height, _screenWidth, _screenHeight);
	GUIUtils::addHighGUIWindow(2, 0, WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE, originalImage.size().width, originalImage.size().height, screenWidth, screenHeight);
	GUIUtils::addHighGUIWindow(0, 1, WINDOW_NAME_CONTRAST_AND_BRIGHTNESS, originalImage.size().width, originalImage.size().height, screenWidth, screenHeight);
	GUIUtils::addHighGUIWindow(1, 1, WINDOW_NAME_TARGET_DETECTION, originalImage.size().width, originalImage.size().height, screenWidth, screenHeight);

	if (optionsOneWindow) {
		namedWindow(WINDOW_NAME_OPTIONS, CV_WINDOW_NORMAL);
		resizeWindow(WINDOW_NAME_OPTIONS, WINDOW_OPTIONS_WIDTH - WINDOW_FRAME_THICKNESS * 2, WINDOW_OPTIONS_HIGHT);
		moveWindow(WINDOW_NAME_OPTIONS, screenWidth - WINDOW_OPTIONS_WIDTH, 0);
	}
	else {
		GUIUtils::addHighGUITrackBarWindow(WINDOW_NAME_BILATERAL_FILTER_OPTIONS, 3, 0, 0, screenWidth);
		GUIUtils::addHighGUITrackBarWindow(WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS, 3, 3, 1, screenWidth);
		GUIUtils::addHighGUITrackBarWindow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS, 2, 6, 2, screenWidth);
	}

	cv::createTrackbar(TRACK_BAR_NAME_BILATERAL_FILTER_DIST, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), imagePreprocessor->getBilateralFilterDistancePtr(), 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BILATERAL_FILTER_COLOR_SIG, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), imagePreprocessor->getBilateralFilterSigmaColorPtr(), 200, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BILATERAL_FILTER_SPACE_SIG, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), imagePreprocessor->getBilateralFilterSigmaSpacePtr(), 200, updateImageAnalysis, (void*)this);

	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_CLIP, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), imagePreprocessor->getClaehClipLimitPtr(), 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_TILE_X, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), imagePreprocessor->getClaehTileXSizePtr(), 20, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_TILE_Y, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), imagePreprocessor->getClaehTileYSizePtr(), 20, updateImageAnalysis, (void*)this);

	cv::createTrackbar(TRACK_BAR_NAME_CONTRAST, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS), imagePreprocessor->getContrastPtr(), 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BRIGHTNESS, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS), imagePreprocessor->getBrightnessPtr(), 1000, updateImageAnalysis, (void*)this);
}