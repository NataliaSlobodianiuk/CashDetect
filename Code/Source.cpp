#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void rearrangeQuadrants(Mat* magnitude);
int matcher(Mat& crop, Scalar avgPixelIntensity);
Mat multiplyInFrequencyDomain(Mat& image, Mat& mask);
Mat magnitude(Mat& first,Mat& second);

int main(int argc, char* argv[]) {

	Mat img = imread("money3.jpg");
	Mat input_image = imread("money3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat cropppedImg;
	Mat clone_img = img.clone();
	RotatedRect minRect;
	Mat M, rotated, rotated1, cropped, cropped1;

	Scalar color = Scalar(255, 255, 255);
	Scalar avgPixelIntensity;

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

	/// Draw contours
	for (int i = 0; i< contours.size(); i++)
	{
		vector<Point> approx;
		double epsilon = 0.075* arcLength(contours[i], true);

		//Approximation to select only the correct contours that can form quadrangle
		approxPolyDP(contours[i], approx, epsilon, true);

		if (approx.size() == 4 && (contourArea(contours[i])>(clone_img.rows*clone_img.cols) / 450) && (contourArea(contours[i])<(clone_img.rows*clone_img.cols) / 130))
		{
			drawContours(img, contours, i, color, CV_FILLED, 8, hierarchy, 0, Point());
			//finding minimum rectangle that contain our contour
			minRect = minAreaRect(Mat(contours[i]));
			Size rect_size = minRect.size;
			if (rect_size.width < rect_size.height)
			{
				swap(rect_size.width, rect_size.height);
				minRect.angle += 90;
			}
			M = getRotationMatrix2D(minRect.center, minRect.angle, 1.0);

			//get minimum rotated rectangle with denomination banknotes
			warpAffine(clone_img, rotated, M, img.size(), INTER_CUBIC);
			getRectSubPix(rotated, rect_size, minRect.center, cropped);

			//get minimum rotated rectangle with contour of denomination banknotes
			warpAffine(img, rotated1, M, img.size(), INTER_CUBIC);
			getRectSubPix(rotated1, rect_size, minRect.center, cropped1);
			cvtColor(cropped1, cropped1, CV_BGR2GRAY);
			cropped1 = cropped1 <254;

			//calculation average R, G, B component in rectangle with denomination banknotes
			avgPixelIntensity = mean(cropped);

			//function that calculate denomination of banknotes
			int nominal = matcher(cropped1, avgPixelIntensity);
			cout << nominal;

		}

	}
	imshow("result", res1);

	waitKey(0);
	return 0;
}

int matcher(Mat& crop, Scalar avgPixelIntensity)
{
	int money_emblem = 0;
	double c = 0;
	bool flag = false;

	//array binary contour emblem of banknotes
	Mat image_array[4];
	image_array[0] = imread("emblem_10.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	image_array[1] = imread("emblem_20.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	image_array[2] = imread("emblem_50.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	image_array[3] = imread("emblem_200.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	//compare input contour with binaries emblem
	for (int i = 0; i < 4; i++)
	{
		c = matchShapes(crop, image_array[i], CV_CONTOURS_MATCH_I1, 0.0);
		if (c < 0.02) flag = true;
	}

	//if input contour owned one of binaries emblem, we can determine denomination by average pixel intensity
	if (flag == true)
	{
		if ((avgPixelIntensity.val[0]>135 && avgPixelIntensity.val[0] < 170) && (avgPixelIntensity.val[1]>85 && avgPixelIntensity.val[1] < 110) && (avgPixelIntensity.val[2]>65 && avgPixelIntensity.val[2] < 90))
		{
			money_emblem = 10;
		}

		if ((avgPixelIntensity.val[0]>145 && avgPixelIntensity.val[0] < 170) && (avgPixelIntensity.val[1]>100 && avgPixelIntensity.val[1] < 115) && (avgPixelIntensity.val[2]>30 && avgPixelIntensity.val[2] < 50))
		{
			money_emblem = 20;
		}

		if ((avgPixelIntensity.val[0]>160 && avgPixelIntensity.val[0] < 180) && (avgPixelIntensity.val[1]>80 && avgPixelIntensity.val[1] < 100) && (avgPixelIntensity.val[2]>30 && avgPixelIntensity.val[2] < 45))
		{
			money_emblem = 50;
		}

		if ((avgPixelIntensity.val[0]>175 && avgPixelIntensity.val[0] < 195) && (avgPixelIntensity.val[1]>100 && avgPixelIntensity.val[1] < 120) && (avgPixelIntensity.val[2]>45 && avgPixelIntensity.val[2] < 63))
		{
			money_emblem = 200;
		}
	}

	return money_emblem;
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

