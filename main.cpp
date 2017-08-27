#include"CurrencyDetection.h"



int main()
{
	cv::Mat image = cv::imread("TestDB\\100-100-200-200__(2).jpg");

	// Rotate and resize image if it is needed
	image = rotateAndResize(image);

	cv::Mat dr;

	double sum = getCurrencySum(image, dr);

	std::cout << sum << std::endl;

	cv::namedWindow("Image", CV_WINDOW_FREERATIO);
	cv::imshow("Image", image);
	cv::waitKey(0);

	cv::namedWindow("Cash detected", CV_WINDOW_FREERATIO);
	cv::imshow("Cash detected", dr);
	cv::waitKey(0);
  	return 0;
}
