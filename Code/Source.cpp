#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void rearrangeQuadrants(Mat* magnitude);
Mat multiplyInFrequencyDomain(Mat& image, Mat& mask);
Mat magnitude(Mat& first,Mat& second);

int main(int argc, char* argv[]) {

	Mat img = imread("money3.jpg");
	Mat input_image = imread("money3.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	//The size of the picture for fast Fourier transform should be 2 in degree
	Mat padded_image;
	Size padded_size(
		getOptimalDFTSize(input_image.cols),
		getOptimalDFTSize(input_image.rows));

	//Creating an image of the optimal size from the input
	copyMakeBorder(input_image, 
		padded_image,
		0,
		padded_size.height - input_image.rows, 
		0, 
		padded_size.width - input_image.cols, 
		BORDER_CONSTANT, 
		Scalar::all(0));

	//Creating a two - channel image in order to create a complex image for the Fourier transform
	Mat planes[] = { Mat_<float>(padded_image), Mat::zeros(padded_size, CV_32F) };
	Mat complex_image;
	merge(planes, 2, complex_image);

	//Direct Fourier transform
	dft(complex_image, complex_image);

	//Create first mask of Sobel(derivative by x)
	Mat xSobel_mask(padded_size, complex_image.type(), Scalar::all(0));

	Rect xSobel_roi(xSobel_mask.cols / 2 - 1, xSobel_mask.rows / 2 - 1, 3, 3);
	Mat xSobel_mask_center(xSobel_mask, xSobel_roi);
	xSobel_mask_center.at<Vec2f>(0, 0)[0] = -1;
	xSobel_mask_center.at<Vec2f>(1, 0)[0] = -2;
	xSobel_mask_center.at<Vec2f>(2, 0)[0] = -1;
	xSobel_mask_center.at<Vec2f>(0, 2)[0] = 1;
	xSobel_mask_center.at<Vec2f>(1, 2)[0] = 2;
	xSobel_mask_center.at<Vec2f>(2, 2)[0] = 1;

	//Mask transformation into frequency domain
	dft(xSobel_mask, xSobel_mask);

	//Filtration in frequency domain
	Mat xSobel_filtered_image = multiplyInFrequencyDomain(complex_image, xSobel_mask);

	//Create second mask of Sobel(derivative by y)
	Mat ySobel_mask(padded_size, complex_image.type(), Scalar::all(0));

	Rect ySobel_roi(ySobel_mask.cols / 2 - 1, ySobel_mask.rows / 2 - 1, 3, 3);
	Mat ySobel_mask_center(ySobel_mask, ySobel_roi);
	ySobel_mask_center.at<Vec2f>(0, 0)[0] = -1;
	ySobel_mask_center.at<Vec2f>(0, 1)[0] = -2;
	ySobel_mask_center.at<Vec2f>(0, 2)[0] = -1;
	ySobel_mask_center.at<Vec2f>(2, 0)[0] = 1;
	ySobel_mask_center.at<Vec2f>(2, 1)[0] = 2;
	ySobel_mask_center.at<Vec2f>(2, 2)[0] = 1;

	//Mask transformation into frequency domain
	dft(ySobel_mask, ySobel_mask);

	//Filtration
	Mat ySobel_filtered_image = multiplyInFrequencyDomain(complex_image, ySobel_mask);

	//The transformation from the frequency domain into spatial
	dft(ySobel_filtered_image, ySobel_filtered_image, DFT_INVERSE | cv::DFT_REAL_OUTPUT);
	dft(xSobel_filtered_image, xSobel_filtered_image, DFT_INVERSE | cv::DFT_REAL_OUTPUT);

	//Rearrange the image so that the bright stuff is in the middle
	rearrangeQuadrants(&xSobel_filtered_image);
	rearrangeQuadrants(&ySobel_filtered_image);

	//Calculation magnitude
	Mat result = magnitude(xSobel_filtered_image, ySobel_filtered_image);

	Mat res1;
	normalize(result, result, 0, 1, CV_MINMAX);
	result.convertTo(res1, CV_8UC1, 255, 0);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	int dilation_size = 1;

	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	threshold(res1, res1, 100, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	for (int i = 0; i < 3; i++)
	{
		dilate(res1, res1, element);
		dilate(res1, res1, element);
		erode(res1, res1, element);
	}
	erode(res1, res1, element);
	erode(res1, res1, element);


	/// Find contours
	findContours(res1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));

	imshow("result", res1);

	waitKey(0);
	return 0;
}


void rearrangeQuadrants(Mat* magnitude) {
	// rearrange the image so that the bright stuff is in the middle
	int center_x = magnitude->cols / 2, center_y = magnitude->rows / 2;

	//get a ROI for each quadrant
	Mat q0(*magnitude, Rect(0, 0, center_x, center_y));   // Top-Left
	Mat q1(*magnitude, Rect(center_x, 0, center_x, center_y));  // Top-Right
	Mat q2(*magnitude, Rect(0, center_y, center_x, center_y));  // Bottom-Left
	Mat q3(*magnitude, Rect(center_x, center_y, center_x, center_y)); // Bottom-Right

	//by rearragning these ROIs it modifies the original image
	// swap top left and bottom right
	Mat temp;
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);

	//swap top right and bottom left
	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);
}

Mat multiplyInFrequencyDomain(Mat& image, Mat& mask) 
{
	Mat result(image.rows, image.cols, CV_32FC2, Scalar::all(0));

	float real_image_element, imagine_image_element, real_mask_element, imagine_mask_element;
	float *image_row_ptr, *mask_row_ptr, *res, *image_channel_ptr, *mask_channel_ptr, *res_channel_ptr;
	for (int i = 0; i < result.rows; i++)
	{
		res = result.ptr<float>(i);
		image_row_ptr = image.ptr<float>(i);
		mask_row_ptr = mask.ptr<float>(i);
		for (int j = 0; j < result.cols; j++)
		{
			image_channel_ptr = image_row_ptr;
			mask_channel_ptr = mask_row_ptr;
			res_channel_ptr = res;
			real_image_element = image_channel_ptr[0];
			imagine_image_element = image_channel_ptr[1];
			real_mask_element = mask_row_ptr[0];
			imagine_mask_element = mask_row_ptr[1];
			res[0] = real_image_element*real_mask_element - imagine_image_element*imagine_mask_element;
			res[1] = real_image_element*imagine_mask_element + imagine_image_element*real_mask_element;
			image_row_ptr += 2;
			mask_row_ptr += 2;
			res += 2;
		}
	}
	return result;
}

Mat magnitude(Mat& first, Mat& second)
{
	//calculate magnitude |z|=sqrt(a*a+b*b) for each element
	float elem_first_matrix, elem_second_matrix;
	float *row_first_matrix, *row_second_matrix, *row_ptr;
	Mat result(first.rows, first.cols, CV_32FC1, cv::Scalar::all(0));

	for (int i = 0; i < result.rows; i++)
	{
		row_ptr = result.ptr<float>(i);
		row_first_matrix = first.ptr<float>(i);
		row_second_matrix = second.ptr<float>(i);

		for (int j = 0; j < result.cols; j++)
		{
			elem_first_matrix = row_first_matrix[j];
			elem_second_matrix = row_second_matrix[j];
			row_ptr[j] = sqrt(elem_first_matrix*elem_first_matrix + elem_second_matrix*elem_second_matrix);
		}

	}
	return result;
}

