#include "CoinsDetection.h"
#include "CurrencyDetection.h"
#include "CommandLineUI.h"

#include <ctime>

int main()
{
	// Start 
	showConsoleHeader();
	int userOption = getUserOption();

	// Filename and image for detection
	string filename;
	Mat image;

	// Read image
	if (userOption != 0) {
		filename = getFilename();
		image = imread(TEST_DB + filename);

		// Check if image is empty
		while (image.empty())
		{
			cout << "  => There is not such image!\n";
			filename = getFilename();
			image = imread(TEST_DB + filename);
		}
	}

	// Select mode
	if (userOption == 1)
	{
		cout << "  >> Recognize currency using SIFT feature transform\n";
		// Rotate and resize image if it is neened
		image = rotateAndResize(image);

		// Detect currency and compute sum
		Mat drawing;
		double sum = getCurrencySum(image, drawing);

		cout << "  >> Sum of currency at the image = " << sum << endl;

		namedWindow("Currency", CV_WINDOW_FREERATIO);
		imshow("Currency", drawing);
		waitKey(0);
	}
	else if (userOption == 2)
	{
		cout << "  >> Recognize currency using Discrete Fourier transform\n";
	}
	else if (userOption == 3)
	{
		cout << "  >> Recognize coins\n";

		Mat drawing;
		image.copyTo(drawing);

		int sum = 0;
		clock_t begin = clock();
		{
			// Detect coins and compute sum
			try
			{
				sum = getCoinsSum(drawing);
			}
			catch (exception e)
			{
				cerr << e.what() << endl;
				waitKey(0);
				return -1;
			}
			namedWindow("Coins", CV_WINDOW_FREERATIO);
			imshow("Coins", drawing);
		}
		clock_t end = clock();

		cout << "  >> Time elapsed: " << (double)(end - begin) / CLOCKS_PER_SEC << " second(s)" << endl;

		cout << "  >> Sum: " << sum / 100 << " hryvnia(s) " << sum % 100 << " hryvnia coins." << endl;

		waitKey(0);
	}
	else if (userOption == 0)
	{
		cout << "	Exit\n";
		waitKey(0);
	}

  	return 0;
}
