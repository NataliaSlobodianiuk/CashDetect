#include"CurrencyDetection.h"
#include"CommandLineUI.h"


int main()
{
	// Start 
	showConsoleHeader();
	int userOption = getUserOption();

	// Filename and image for detection
	std::string filename;
	cv::Mat image;

	// Read image
	if (userOption != 0) {
		filename = getFilename();
		image = cv::imread(TEST_DB + filename);

		// Check if image is empty
		while (image.empty())
		{
			std::cout << "  => There is not such image!\n";
			filename = getFilename();
			image = cv::imread(TEST_DB + filename);
		}
	}

	// Select mode
	if (userOption == 1)
	{
		std::cout << "  >> Recognize currency using SIFT feature transform\n";
		// Rotate and resize image if it is neened
		image = rotateAndResize(image);

		// Detect currency and compute sum
		cv::Mat drawing;
		double sum = getCurrencySum(image, drawing);

		std::cout << "  >> Sum of currency at the image = " << sum << std::endl;

		cv::namedWindow("Currency", CV_WINDOW_FREERATIO);
		cv::imshow("Currency", drawing);
		cv::waitKey(0);
	}
	else if (userOption == 2)
	{
		std::cout << "  >> Recognize currency using Discrete Fourier transform\n";
	}
	else if (userOption == 3)
	{
		std::cout << "  >> Recognize coins\n";
	}
	else if (userOption == 0)
	{
		std::cout << "	Exit\n";
	}
  	return 0;
}
