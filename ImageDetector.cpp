#include "ImageDetector.h"


ImageDetector::ImageDetector(Ptr<FeatureDetector> _featureDetector, Ptr<DescriptorExtractor> _descriptorExtractor, Ptr<DescriptorMatcher> _descriptorMatcher, Ptr<ImagePreprocessor> _imagePreprocessor,
	const vector<string>& _referenceImagesDirectories,
	bool _useInliersGlobalMatch,
	const string& _referenceImagesListPath, const string& _testImagesListPath) :
	featureDetector(_featureDetector), descriptorExtractor(_descriptorExtractor), descriptorMatcher(_descriptorMatcher),
	imagePreprocessor(_imagePreprocessor),
	referenceImagesDirectories(_referenceImagesDirectories), referenceImagesListPath(_referenceImagesListPath), testImagesListPath(_testImagesListPath),
	contourAspectRatioRange(-1, -1), contourCircularityRange(-1, -1) {

	// setup DB of currency
	setupTargetDB(referenceImagesListPath, _useInliersGlobalMatch);
	setupTargetsShapesRanges();
}


ImageDetector::~ImageDetector() {}


bool ImageDetector::setupTargetDB(const string& referenceImagesListPath, bool useInliersGlobalMatch) {
	targetDetectors.clear();

	ifstream imgsList(referenceImagesListPath);

	if (imgsList.is_open()) {
		string configurationLine;
		vector<string> configurations;
		while (getline(imgsList, configurationLine)) {
			configurations.push_back(configurationLine);
		}
		int numberOfFiles = configurations.size();

		// make some parallel code
		#pragma omp parallel for schedule(dynamic)
		for (int configIndex = 0; configIndex < numberOfFiles; ++configIndex) {
			string filename;
			size_t targetTag;
			string separator;
			stringstream ss(configurations[configIndex]);
			ss >> filename >> separator >> targetTag >> separator;

			TargetDetector targetDetector(featureDetector, descriptorExtractor, descriptorMatcher, targetTag, useInliersGlobalMatch);

			for (size_t i = 0; i < referenceImagesDirectories.size(); ++i) {
				string referenceImagesDirectory = referenceImagesDirectories[i];
				Mat targetImage;

				// Create file path to image for DB
				stringstream referenceImgePath;
				referenceImgePath << REFERENCE_IMGAGES_DIRECTORY << referenceImagesDirectory << "/" << filename;
				
				// Load image into targetImage
				if (imagePreprocessor->loadAndPreprocessImage(referenceImgePath.str(), targetImage, CV_LOAD_IMAGE_GRAYSCALE, false)) {
					string filenameWithoutExtension = ImageUtils::getFilenameWithoutExtension(filename);

					// create filename for mask
					stringstream maskFilename;
					maskFilename << REFERENCE_IMGAGES_DIRECTORY << referenceImagesDirectory << "/" << filenameWithoutExtension << MASK_TOKEN << MASK_EXTENSION;

					// load mask 
					Mat targetROIs;
					if (ImageUtils::loadBinaryMask(maskFilename.str(), targetROIs)) {
						targetDetector.setupTargetRecognition(targetImage, targetROIs);

						vector<KeyPoint>& targetKeypoints = targetDetector.getTargetKeypoints();
						
						// writing results of keypoints detector into file
						/*stringstream imageKeypointsFilename;
						imageKeypointsFilename << REFERENCE_IMGAGES_ANALYSIS_DIRECTORY << filenameWithoutExtension << "_" << referenceImagesDirectory << selectorTags << IMAGE_OUTPUT_EXTENSION;
						if (targetKeypoints.empty()) {
							imwrite(imageKeypointsFilename.str(), targetImage);
						}
						else {
							Mat imageKeypoints;
							cv::drawKeypoints(targetImage, targetKeypoints, imageKeypoints, TARGET_KEYPOINT_COLOR);
							imwrite(imageKeypointsFilename.str(), imageKeypoints);
						}*/
					}
				}
			}

			#pragma omp critical
			targetDetectors.push_back(targetDetector);
		}
		return !targetDetectors.empty();
	}
	else {
		return false;
	}
}


void ImageDetector::setupTargetsShapesRanges(const string& maskPath) {
	Mat shapeROIs;
	if (ImageUtils::loadBinaryMask(maskPath, shapeROIs)) {
		vector< vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(shapeROIs, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		int contoursSize = (int)contours.size();

#pragma omp parallel for
		for (int i = 0; i < contoursSize; ++i) {
			double contourAspectRatio = ImageUtils::computeContourAspectRatio(contours[i]);
			double contourCircularity = ImageUtils::computeContourCircularity(contours[i]);

#pragma omp critical
			if (contourAspectRatioRange[0] == -1 || contourAspectRatio < contourAspectRatioRange[0]) {
				contourAspectRatioRange[0] = contourAspectRatio;
			}

#pragma omp critical
			if (contourAspectRatioRange[1] == -1 || contourAspectRatio > contourAspectRatioRange[1]) {
				contourAspectRatioRange[1] = contourAspectRatio;
			}

#pragma omp critical
			if (contourCircularityRange[0] == -1 || contourCircularity < contourCircularityRange[0]) {
				contourCircularityRange[0] = contourCircularity;
			}

#pragma omp critical
			if (contourCircularityRange[1] == -1 || contourCircularity > contourCircularityRange[1]) {
				contourCircularityRange[1] = contourCircularity;
			}
		}
	}
}


Ptr< vector< Ptr<DetectorResult> > > ImageDetector::detectTargets(Mat& image, float minimumMatchAllowed, float minimumTargetAreaPercentage,
	float maxDistanceRatio, float reprojectionThresholdPercentage, double confidence, int maxIters, size_t minimumNumberInliers) {
	Ptr< vector< Ptr<DetectorResult> > > detectorResults(new vector< Ptr<DetectorResult> >());

	vector<KeyPoint> keypointsQueryImage;
	featureDetector->detect(image, keypointsQueryImage);
	if (keypointsQueryImage.size() < 4) { return detectorResults; }

	Mat descriptorsQueryImage;
	descriptorExtractor->compute(image, keypointsQueryImage, descriptorsQueryImage);

	cv::drawKeypoints(image, keypointsQueryImage, image, NONTARGET_KEYPOINT_COLOR);

	float bestMatch = 0;
	Ptr<DetectorResult> bestDetectorResult;

	int targetDetectorsSize = targetDetectors.size();
	bool validDetection = true;
	float reprojectionThreshold = image.cols * reprojectionThresholdPercentage;
	//float reprojectionThreshold = 3.0;

	do {
		bestMatch = 0;
		// make some parallel loop
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < targetDetectorsSize; ++i) {
			targetDetectors[i].updateCurrentLODIndex(image);
			Ptr<DetectorResult> detectorResult = targetDetectors[i].analyzeImage(keypointsQueryImage, descriptorsQueryImage, maxDistanceRatio, reprojectionThreshold, confidence, maxIters, minimumNumberInliers);
			if (detectorResult->getBestROIMatch() > minimumMatchAllowed) {
				float contourArea = (float)cv::contourArea(detectorResult->getTargetContour());
				float imageArea = (float)(image.cols * image.rows);
				float contourAreaPercentage = contourArea / imageArea;

				if (contourAreaPercentage > minimumTargetAreaPercentage) {
					double contourAspectRatio = ImageUtils::computeContourAspectRatio(detectorResult->getTargetContour());
					if (contourAspectRatio > contourAspectRatioRange[0] && contourAspectRatio < contourAspectRatioRange[1]) {
						double contourCircularity = ImageUtils::computeContourCircularity(detectorResult->getTargetContour());
						if (contourCircularity > contourCircularityRange[0] && contourCircularity < contourCircularityRange[1]) {
							if (cv::isContourConvex(detectorResult->getTargetContour())) {
								#pragma omp critical
								{
									if (detectorResult->getBestROIMatch() > bestMatch) {
										bestMatch = detectorResult->getBestROIMatch();
										bestDetectorResult = detectorResult;
									}
								}
							}
						}
					}
				}
			}
		}

		validDetection = bestMatch > minimumMatchAllowed && bestDetectorResult->getInliers().size() > minimumNumberInliers;

		if (bestDetectorResult != NULL && validDetection) {
			detectorResults->push_back(bestDetectorResult);

			// remove inliers of best match to detect more occurrences of targets
			ImageUtils::removeInliersFromKeypointsAndDescriptors(bestDetectorResult->getInliers(), keypointsQueryImage, descriptorsQueryImage);
		}
	} while (validDetection);

	return detectorResults;
}


vector<size_t> ImageDetector::detectTargetsAndOutputResults(Mat& image, const string& imageFilename, bool useHighGUI) {
	Mat imageBackup = image.clone();
	Ptr< vector< Ptr<DetectorResult> > > detectorResultsOut = detectTargets(image);
	vector<size_t> results;

	for (size_t i = 0; i < detectorResultsOut->size(); ++i) {
		Ptr<DetectorResult> detectorResult = (*detectorResultsOut)[i];
		results.push_back(detectorResult->getTargetValue());

		cv::drawKeypoints(image, detectorResult->getInliersKeypoints(), image, TARGET_KEYPOINT_COLOR);

		stringstream ss;
		ss << detectorResult->getTargetValue();

		Mat imageMatchesSingle = imageBackup.clone();
		Mat matchesInliers = detectorResult->getInliersMatches(imageMatchesSingle);

		try {
			Rect boundingBox = cv::boundingRect(detectorResult->getTargetContour());
			ImageUtils::correctBoundingBox(boundingBox, image.cols, image.rows);
			GUIUtils::drawLabelInCenterOfROI(ss.str(), image, boundingBox);
			GUIUtils::drawLabelInCenterOfROI(ss.str(), matchesInliers, boundingBox);
			ImageUtils::drawContour(image, detectorResult->getTargetContour(), cv::Scalar(0, 255, 0));
			ImageUtils::drawContour(matchesInliers, detectorResult->getTargetContour(), cv::Scalar(0, 255, 0));
		}
		catch (...) {
			std::cerr << "!!! Drawing outside image !!!" << endl;
		}

		if (useHighGUI) {
			stringstream windowName;
			windowName << "Target inliers matches (window " << i << ")";
			cv::namedWindow(windowName.str(), CV_WINDOW_KEEPRATIO);
			cv::imshow(windowName.str(), matchesInliers);
			cv::waitKey(10);
		}
	}

	sort(results.begin(), results.end());

	cout << "    -> Detected " << results.size() << (results.size() != 1 ? " targets" : " target");
	size_t globalResult = 0;
	stringstream resultsSS;
	if (!results.empty()) {
		resultsSS << " (";
		for (size_t i = 0; i < results.size(); ++i) {
			size_t resultValue = results[i];
			resultsSS << " " << resultValue;
			globalResult += resultValue;
		}
		resultsSS << " )";
		cout << resultsSS.str();
	}
	cout << endl;

	stringstream globalResultSS;
	globalResultSS << "Global result: " << globalResult << resultsSS.str();
	Rect globalResultBoundingBox(0, 0, image.cols, image.rows);
	GUIUtils::drawImageLabel(globalResultSS.str(), image, globalResultBoundingBox);
	return results;
}


DetectorEvaluationResult ImageDetector::evaluateDetector(const string& testImgsList, bool saveResults) {
	double globalPrecision = 0;
	double globalRecall = 0;
	double globalAccuracy = 0;
	size_t numberTestImages = 0;
	ifstream imgsList(testImgsList);
	if (imgsList.is_open()) {
		string filename;
		vector<string> imageFilenames;
		vector< vector<size_t> > expectedResults;
		while (getline(imgsList, filename)) {
			imageFilenames.push_back(filename);

			vector<size_t> expectedResultFromTest;
			extractExpectedResultsFromFilename(filename, expectedResultFromTest);
			expectedResults.push_back(expectedResultFromTest);
		}
		int numberOfTests = imageFilenames.size();

		cout << "    -> Evaluating detector with " << numberOfTests << " test images..." << endl;
		PerformanceTimer globalPerformanceTimer;
		globalPerformanceTimer.start();

		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < numberOfTests; ++i) {
			PerformanceTimer testPerformanceTimer;
			testPerformanceTimer.start();

			string imageFilename = imageFilenames[i];
			//string imageFilename = ImageUtils::getFilenameWithoutExtension("");
			string imageFilenameWithPath = TEST_IMGAGES_DIRECTORY + imageFilenames[i];
			stringstream detectorEvaluationResultSS;
			DetectorEvaluationResult detectorEvaluationResult;
			Mat imagePreprocessed;
			cout << "\n    -> Evaluating image " << imageFilename << " (" << (i + 1) << "/" << numberOfTests << ")" << endl;
			if (imagePreprocessor->loadAndPreprocessImage(imageFilenameWithPath, imagePreprocessed, CV_LOAD_IMAGE_GRAYSCALE, false)) {
				vector<size_t> results = detectTargetsAndOutputResults(imagePreprocessed, imageFilename, false);

				detectorEvaluationResult = DetectorEvaluationResult(results, expectedResults[i]);
				globalPrecision += detectorEvaluationResult.getPrecision();
				globalRecall += detectorEvaluationResult.getRecall();
				globalAccuracy += detectorEvaluationResult.getAccuracy();

				detectorEvaluationResultSS << PRECISION_TOKEN << ": " << detectorEvaluationResult.getPrecision() << " | " << RECALL_TOKEN << ": " << detectorEvaluationResult.getRecall() << " | " << ACCURACY_TOKEN << ": " << detectorEvaluationResult.getAccuracy();

				++numberTestImages;
			}
			cout << "    -> Evaluation of image " << imageFilename << " finished in " << testPerformanceTimer.getElapsedTimeFormated() << endl;
			cout << "    -> " << detectorEvaluationResultSS.str() << endl;
		}

		globalPrecision /= (double)numberTestImages;
		globalRecall /= (double)numberTestImages;
		globalAccuracy /= (double)numberTestImages;

		stringstream detectorEvaluationGloablResultSS;
		detectorEvaluationGloablResultSS << GLOBAL_PRECISION_TOKEN << ": " << globalPrecision << " | " << GLOBAL_RECALL_TOKEN << ": " << globalRecall << " | " << GLOBAL_ACCURACY_TOKEN << ": " << globalAccuracy;
		cout << "\n    -> Finished evaluation of detector in " << globalPerformanceTimer.getElapsedTimeFormated() << " || " << detectorEvaluationGloablResultSS.str() << "\n" << endl;
	}

	return DetectorEvaluationResult(globalPrecision, globalRecall, globalAccuracy);
}


void ImageDetector::extractExpectedResultsFromFilename(string filename, vector<size_t>& expectedResultFromTestOut) {
	for (size_t i = 0; i < filename.size(); ++i) {
		char letter = filename[i];
		if (letter == '-') {
			filename[i] = ' ';
		}
		else if (letter == '.' || letter == '_') {
			filename = filename.substr(0, i);
			break;
		}
	}

	stringstream ss(filename);
	size_t number;
	while (ss >> number) {
		expectedResultFromTestOut.push_back(number);
	}
}