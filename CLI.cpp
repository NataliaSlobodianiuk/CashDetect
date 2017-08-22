#include "CLI.h"

void CLI::showConsoleHeader() {
	cout << "####################################################################################################\n";
	cout << "  >>>                                    Currency recognition                                  <<<  \n";
	cout << "####################################################################################################\n\n";
}


void CLI::startInteractiveCLI() {
	int userOption = 1;
	string filename = "";

	ConsoleInput::getInstance()->clearConsoleScreen();
	showConsoleHeader();

	int screenWidth = 1920; // ConsoleInput::getInstance()->getIntCin("  >> Screen width (used to arrange windows): ", "  => Width >= 100 !!!\n", 100);
	int screenHeight = 1080; // ConsoleInput::getInstance()->getIntCin("  >> Screen height (used to arrange windows): ", "  => Width >= 100 !!!\n", 100);
	bool optionsOneWindow = false; // ConsoleInput::getInstance()->getYesNoCin("  >> Use only one window for options trackbars? (Y/N): ");
	bool setupOfImageRecognitionDone = false;

	do {
		try {
			ConsoleInput::getInstance()->clearConsoleScreen();
			showConsoleHeader();

			if (setupOfImageRecognitionDone) {
				userOption = getUserOption();
				if (userOption == 1) {
					setupImageRecognition();
				}
				else if (userOption == 2) {
					imageDetector->evaluateDetector(TEST_IMGAGES_LIST);
				}
				else {
					if (userOption == 3) {
						filename = "";
						do {
							cout << "  >> Path to file inside imgs\\testDB folder: ";
							filename = ConsoleInput::getInstance()->getLineCin();

							if (filename == "") {
								cerr << "  => File path can't be empty!\n" << endl;
							}
						} while (filename == "");
					}
					ImageAnalysis imageAnalysis(imagePreprocessor, imageDetector);
					imageAnalysis.setScreenWidth(screenWidth);
					imageAnalysis.setScreenHeight(screenHeight);
					imageAnalysis.setOptionsOneWindow(optionsOneWindow);

					switch (userOption) {
					case 3: { if (!imageAnalysis.processImage(filename, true)) { cerr << "  => Failed to load image " << filename << "!" << endl; } break; }
					default: break;
					}
				}
			}
			else {
				setupImageRecognition();
				setupOfImageRecognitionDone = true;
			}

			if (userOption != 0) {
				cout << "\n\n" << endl;
				ConsoleInput::getInstance()->getUserInput();
			}
		}
		catch (...) {
			cerr << "\n\n\n!!!!! Caught unexpected exception !!!!!\n\n\n" << endl;
		}
	} while (userOption != 0);
	ConsoleInput::getInstance()->getUserInput();
}


int CLI::getUserOption() {
	cout << " ## Menu:\n";
	cout << "   1 - Setup image recognition configuration\n";
	cout << "   2 - Evaluate detector\n";
	cout << "   3 - Test detector from image\n";
	cout << "   0 - Exit\n";

	return ConsoleInput::getInstance()->getIntCin("\n >>> Option [0, 3]: ", "Select one of the options above!", 0, 4);
}


void CLI::setupImageRecognition() {
	cout << "\n\n ## Image recognition setup:\n" << endl;
	int featureDetectorSelection = selectFeatureDetector();
	cout << "\n\n\n";
	int descriptorExtractorSelection = selectDescriptorExtractor();
	cout << "\n\n\n";
	int descriptorMatcherSelection = selectDescriptorMatcher();
	cout << "\n\n\n";

	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorExtractor> descriptorExtractor;
	Ptr<DescriptorMatcher> descriptorMatcher;

	switch (featureDetectorSelection) {
	case 1: { featureDetector = cv::xfeatures2d::SiftFeatureDetector::create();			 
			break; }
	case 2: { featureDetector = cv::xfeatures2d::SURF::create(400);		 
		break; }
	case 3: { featureDetector = cv::GFTTDetector::create();	
		break; }
	case 4: { featureDetector = cv::FastFeatureDetector::create();			
		break; }
	case 5: { featureDetector = cv::ORB::create();			 
		break; }
	case 6: { featureDetector = cv::BRISK::create();						
		break; }
	case 7: { featureDetector = cv::xfeatures2d::StarDetector::create();			
		break; }
	case 8: { featureDetector = cv::MSER::create();			 
		break; }
	default: break;
	}

	switch (descriptorExtractorSelection) {
	case 1: { descriptorExtractor = cv::xfeatures2d::SiftDescriptorExtractor::create();	
		break; }
	case 2: { descriptorExtractor = cv::xfeatures2d::SurfDescriptorExtractor::create();	
		break; }
	case 3: { descriptorExtractor = cv::xfeatures2d::FREAK::create();					
		break; }
	case 4: { descriptorExtractor = cv::xfeatures2d::BriefDescriptorExtractor::create();	 
		break; }
	case 5: { descriptorExtractor = cv::ORB::create();	
			break; }
	case 6: { descriptorExtractor = cv::BRISK::create();					 
				break; }
	default: break;
	}

	int bfNormType;
	Ptr<cv::flann::IndexParams> flannIndexParams/* = new cv::flann::AutotunedIndexParams()*/;
	if (descriptorExtractorSelection > 2) { // binary descriptors		
		bfNormType = cv::NORM_HAMMING;
		//flannIndexParams = new cv::flann::HierarchicalClusteringIndexParams();
		flannIndexParams = new cv::flann::LshIndexParams(12, 20, 2);
	}
	else { // float descriptors		
		bfNormType = cv::NORM_L2;
		flannIndexParams = new cv::flann::KDTreeIndexParams();
	}

	switch (descriptorMatcherSelection) {
		case 1: { descriptorMatcher = new cv::FlannBasedMatcher(flannIndexParams);	
				break; }
	case 2: { descriptorMatcher = new cv::BFMatcher(bfNormType, false);			
			break; }
	default: break;
	}

	// set DB for recognition 
	// in this case low quality DB is used: 512 X 270
	vector<string> imagesDBLevelOfDetail;
	imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_VERY_LOW);
	imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_LOW);
	imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_MEDIUM);

	// set match
	// in this case localMatch is used 
	// matcher use masks with important currency regions 
	bool inliersSelectionMethodFlagToUseGlobalMatch = true;	

	// create image detector
	imageDetector = new ImageDetector(featureDetector, descriptorExtractor, descriptorMatcher, imagePreprocessor, imagesDBLevelOfDetail, inliersSelectionMethodFlagToUseGlobalMatch);
}


int CLI::selectFeatureDetector() {
	cout << "  => Select feature detector:\n";
	cout << "    1 - SIFT\n";
	cout << "    2 - SURF\n";
	cout << "    3 - GFTT\n";
	cout << "    4 - FAST\n";
	cout << "    5 - ORB\n";
	cout << "    6 - BRISK\n";
	cout << "    7 - STAR\n";
	cout << "    8 - MSER\n";

	return ConsoleInput::getInstance()->getIntCin("\n >>> Option [1, 7]: ", "Select one of the options above!", 1, 9);
}


int CLI::selectDescriptorExtractor() {
	cout << "  => Select descriptor extractor:\n";
	cout << "    1 - SIFT\n";
	cout << "    2 - SURF\n";
	cout << "    3 - FREAK\n";
	cout << "    4 - BRIEF\n";
	cout << "    5 - ORB\n";
	cout << "    6 - BRISK\n";

	return ConsoleInput::getInstance()->getIntCin("\n >>> Option [1, 6]: ", "Select one of the options above!", 1, 7);
}


int CLI::selectDescriptorMatcher() {
	cout << "  => Select descriptor matcher:\n";
	cout << "    1 - FlannBasedMatcher\n";
	cout << "    2 - BFMatcher\n";

	return ConsoleInput::getInstance()->getIntCin("\n >>> Option [1, 2]: ", "Select one of the options above!", 1, 3);
}

