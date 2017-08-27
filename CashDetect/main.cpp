#include "CoinsDetection.h"

#include <ctime>
#include <exception>
#include <iostream>

int main(int argc, char **argv)
{
	Mat src;

	for (int i = 1; i < argc; i++)
	{
		src = imread(argv[i]);

		if (!src.data)
			return -1;

		int sum = 0;
		clock_t begin = clock();
		{
			int height = src.rows;
			int width = src.cols;

			if (height > width)
			{
				rotate(src, src, ROTATE_90_CLOCKWISE);
			}
			namedWindow("Source", CV_WINDOW_FREERATIO);
			imshow("Source", src);

			try
			{
				src = getA4(src);
			}
			catch (exception e)
			{
				cout << e.what() << endl;
				return -1;
			}

			height = src.rows;
			width = src.cols;

			if (height > width)
			{
				rotate(src, src, ROTATE_90_CLOCKWISE);
			}
			namedWindow("SourceA4", CV_WINDOW_FREERATIO);
			imshow("SourceA4", src);

			sum = getCoinsSum(src);
			namedWindow("Result", CV_WINDOW_FREERATIO);
			imshow("Result", src);
		}
		clock_t end = clock();

		cout << "Time elapsed: " << (double)(end - begin) / CLOCKS_PER_SEC << " second(s)" << endl;

		cout << "Sum: " << sum / 100 << " hryvnia(s) " << sum % 100 << " hryvnia coins." << endl;

		int key = waitKey(0);

		if (key == 27)
		{
			return 0;
		}
	}

	return 0;
}
