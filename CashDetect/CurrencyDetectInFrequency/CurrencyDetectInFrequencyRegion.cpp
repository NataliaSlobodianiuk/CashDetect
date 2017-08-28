#include "CurrencyDetectInFrequencyRegion.h"

#include <cmath>
#include <string>

Mat filtering_image(string path)
{
	Mat img = imread(path);
	Mat color_img = imread(path);
	Mat input_image = imread("path", CV_LOAD_IMAGE_GRAYSCALE);
	int nominal;

	Mat cropppedImg;
	Mat clone_img = img.clone();

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
	Mat res2 = res1.clone();

	medianBlur(res1, res1, 7);
	Mat padded_image_from_magnitude;
	Size padded_size_from_magnitude(
		getOptimalDFTSize(res1.cols),
		getOptimalDFTSize(res1.rows));
	//Creating an image of the optimal size from the input
	copyMakeBorder(res1,
		padded_image_from_magnitude,
		0,
		padded_size_from_magnitude.height - res1.rows,
		0,
		padded_size_from_magnitude.width - res1.cols,
		BORDER_CONSTANT,
		Scalar::all(0));


	//Creating a two - channel image in order to create a complex image for the Fourier transform
	Mat planes1[] = { Mat_<float>(padded_image_from_magnitude), Mat::zeros(padded_size_from_magnitude, CV_32F) };
	Mat complex_image_magnitude;
	merge(planes1, 2, complex_image_magnitude);

	//Direct Fourier transform
	dft(complex_image_magnitude, complex_image_magnitude);

	int dilation_size = 1;

	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));

	//Mask  
	//		0 2 0 2 0
	//		2 0 2 0 2
	//		0 2 0 2 0
	//		2 0 2 0 2
	//		0 2 0 2 0
	//Mask to increase the intensity of the borders
	Mat mask(padded_size, complex_image.type(), Scalar::all(0));

	Rect mask_roi(mask.cols / 2 - 2, mask.rows / 2 - 2, 5, 5);
	Mat mask_center(mask, mask_roi);
	mask_center.at<Vec2f>(0, 1)[0] = 2;
	mask_center.at<Vec2f>(0, 3)[0] = 2;
	mask_center.at<Vec2f>(1, 0)[0] = 2;
	mask_center.at<Vec2f>(1, 2)[0] = 2;
	mask_center.at<Vec2f>(1, 4)[0] = 2;
	mask_center.at<Vec2f>(2, 1)[0] = 2;
	mask_center.at<Vec2f>(2, 3)[0] = 2;
	mask_center.at<Vec2f>(3, 0)[0] = 2;
	mask_center.at<Vec2f>(3, 2)[0] = 2;
	mask_center.at<Vec2f>(3, 4)[0] = 2;
	mask_center.at<Vec2f>(4, 1)[0] = 2;
	mask_center.at<Vec2f>(4, 3)[0] = 2;

	//Mask transformation into frequency domain
	dft(mask, mask);

	//Filtration in frequency domain
	Mat mask_filtered_image = multiplyInFrequencyDomain(complex_image_magnitude, mask);

	//The transformation from the frequency domain into spatial
	dft(mask_filtered_image, mask_filtered_image, DFT_INVERSE | cv::DFT_REAL_OUTPUT);

	Mat resmask;
	//Rearrange the image so that the bright stuff is in the middle
	rearrangeQuadrants(&mask_filtered_image);
	normalize(mask_filtered_image, mask_filtered_image, 0, 1, CV_MINMAX);
	mask_filtered_image.convertTo(resmask, CV_8UC1, 255, 0);
	return resmask;
}

int matcher(Mat& crop)
{
	int money_emblem = 0;
	Scalar color = mean(crop);
	//some bankcotes emblem have different ratio cols to rows
	if ((double(crop.rows)) / (double(crop.cols)) < 0.7)
	{
		if ((color.val[0] > 160 && color.val[0] < 190) && (color.val[1]>150 && color.val[1] < 170) && (color.val[2]>110 && color.val[2] < 140))
		{
			money_emblem = 1;
		}
		if ((color.val[0] > 140 && color.val[0] < 170) && (color.val[1]>140 && color.val[1] < 165) && (color.val[2]>145 && color.val[2] < 170))
		{
			money_emblem = 200;
		}
	}

	if (((double(crop.rows)) / (double(crop.cols)) > 0.7) && ((double(crop.rows)) / (double(crop.cols)) < 1.3))
	{
		if ((color.val[0] > 135 && color.val[0] < 160) && (color.val[1]>130 && color.val[1] < 150) && (color.val[2]>120 && color.val[2] < 140))
		{
			money_emblem = 5;
		}
		if ((color.val[0] > 130 && color.val[0] < 150) && (color.val[1]>130 && color.val[1] < 150) && (color.val[2]>160 && color.val[2] < 183))
		{
			money_emblem = 10;
		}
		if ((color.val[0] > 120 && color.val[0] < 145) && (color.val[1]>120 && color.val[1] < 145) && (color.val[2]>140 && color.val[2] < 160))
		{
			money_emblem = 2;
		}
		if ((color.val[0] > 130 && color.val[0] < 150) && (color.val[1]>145 && color.val[1] < 165) && (color.val[2]>140 && color.val[2] < 160))
		{
			money_emblem = 20;
		}
	}

	return money_emblem;
}


Mat find_all_contours(Mat& resmask, string path)
{
	RotatedRect minRect;
	Mat M, rotated, rotated1, cropped, cropped1;
	vector<vector<Point>> contours;
	int dilation_size = 1;

	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	Size s(resmask.cols, resmask.rows);
	Mat color_img = imread(path);
	//Transformation that help find contours
	adaptiveThreshold(resmask, resmask, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 75, 0);
	resize(resmask, resmask, Size(resmask.cols / 3, resmask.rows / 3));
	resize(resmask, resmask, s);
	findContours(resmask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));

	//additional image
	Mat zero = Mat::zeros(resmask.rows, resmask.cols, CV_8UC1);
	Mat good_zero_contours = Mat::zeros(resmask.rows, resmask.cols, CV_8UC1);

	for (int i = 0; i < contours.size(); i++)
	{
		//Check the contour by size
		if ((contourArea(contours[i])>(resmask.rows*resmask.cols) / 3000) && (contourArea(contours[i]) < (resmask.rows*resmask.cols) / 200))
		{
			minRect = minAreaRect(Mat(contours[i]));
			Size rect_size = minRect.size;
			if (rect_size.width < rect_size.height)
			{
				swap(rect_size.width, rect_size.height);
				minRect.angle += 90;
			}
			M = getRotationMatrix2D(minRect.center, minRect.angle, 1.0);

			//get minimum rotated rectangle with denomination banknotes
			warpAffine(color_img, rotated, M, color_img.size(), INTER_CUBIC);
			getRectSubPix(rotated, rect_size, minRect.center, cropped);

			//filtered contours by ratio cols to rows
			if ((double(cropped.rows) / double(cropped.cols)) < 3 && (double(cropped.rows) / double(cropped.cols)) > 0.3333)
			{
				drawContours(zero, contours, i, Scalar(255), -1);
			}

		}
	}

	dilate(zero, zero, element);
	dilate(zero, zero, element);
	return zero;
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

Mat find_result_contours(Mat& zero, string path)
{
	int nominal = 0;
	RotatedRect minRect;
	Mat M, rotated, rotated1, cropped, cropped1;
	Mat img = imread(path);
	Mat good_zero_contours = Mat::zeros(img.cols, img.rows, CV_8UC1);
	Mat clone_zero = zero.clone();
	vector<vector<Point>> contours_zero_image;

	//find good contours, because on the previous photo has small contours that may distort important contours
	findContours(zero, contours_zero_image, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));
	for (int i = 0; i < contours_zero_image.size(); i++)
	{
		if ((contourArea(contours_zero_image[i])>(img.rows*img.cols) / 3000) && (contourArea(contours_zero_image[i]) < (img.rows*img.cols) / 200))
		{
			minRect = minAreaRect(Mat(contours_zero_image[i]));
			Size rect_size = minRect.size;
			if (rect_size.width < rect_size.height)
			{
				swap(rect_size.width, rect_size.height);
				minRect.angle += 90;
			}
			M = getRotationMatrix2D(minRect.center, minRect.angle, 1.0);

			//get minimum rotated rectangle with denomination banknotes
			warpAffine(clone_zero, rotated, M, img.size(), INTER_CUBIC);
			getRectSubPix(rotated, rect_size, minRect.center, cropped);

			warpAffine(img, rotated, M, img.size(), INTER_CUBIC);
			getRectSubPix(rotated, rect_size, minRect.center, cropped1);

			//rejecting the wrong contours
			double count_white = countNonZero(cropped);
			double count_black = cropped.cols * cropped.rows - count_white;
			if ((count_white / count_black)>2)
			{
				drawContours(good_zero_contours, contours_zero_image, i, Scalar(255), -1);
				nominal = matcher(cropped);
				if (nominal != 0)
				{
					putText(img, to_string(nominal), minRect.center, FONT_HERSHEY_PLAIN, 3.0, CV_RGB(0, 255, 0), 2.0);
				}
			}
		}

	}
	return img;
}