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
	int bfNormType = cv::NORM_L2;
	Ptr<cv::xfeatures2d::SiftFeatureDetector> featureDetector = cv::xfeatures2d::SiftFeatureDetector::create();
	Ptr<cv::xfeatures2d::SiftDescriptorExtractor> descriptorExtractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
	Ptr<cv::BFMatcher> descriptorMatcher  = new cv::BFMatcher(bfNormType, false);

	// set DB for recognition 
	// in this case all quality DB is used: 256 * 137; 512 * 270; 772 * 405
	vector<string> imagesDBLevelOfDetail;
	imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_VERY_LOW);
	imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_LOW);
	imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_MEDIUM);

	// set match
	// in this case globalMatch is used 
	// matcher use masks with important currency regions 
	bool inliersSelectionMethodFlagToUseGlobalMatch = true;	

	// create image detector
	imageDetector = new ImageDetector(featureDetector, descriptorExtractor, descriptorMatcher, imagePreprocessor, imagesDBLevelOfDetail, inliersSelectionMethodFlagToUseGlobalMatch);
}


