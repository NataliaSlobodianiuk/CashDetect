#include "CoinsDetection\CoinsDetection.h"
#include "CurrencyDetection\CurrencyDetection.h"
#include "UI\CommandLineUI.h"

#include <ctime>

//defines for winnames
#define COINS_SCR "Coins Source"
#define COINS_RESULT "Coins Result"
#define CURRENCY "Currency"

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

		namedWindow(CURRENCY, CV_WINDOW_FREERATIO);
		imshow(CURRENCY, drawing);
		waitKey(0);
	}
	else if (userOption == 2)
	{
		cout << "  >> Recognize coins\n";

		namedWindow(COINS_SCR, CV_WINDOW_FREERATIO);
		imshow(COINS_SCR, image);

		//Making copy of the image in order not to draw anything
		//on the input photo
		Mat drawing;
		image.copyTo(drawing);

		int sum = 0;
		clock_t begin = clock();
		{
			try
			{
				// Detect coins, draw them on the image and compute the sum
				// Throw exception if the program doesn't find an A4 contour
				sum = calcCoinsSum(drawing);
			}
			catch (exception e)
			{
				cerr << e.what() << endl;
				waitKey(0);
				return -1;
			}
			namedWindow(COINS_RESULT, CV_WINDOW_FREERATIO);
			imshow(COINS_RESULT, drawing);
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
