#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void rearrangeQuadrants(Mat* magnitude);
int matcher(Mat& crop);
Mat multiplyInFrequencyDomain(Mat& image, Mat& mask);
Mat magnitude(Mat& first, Mat& second);
Mat filtering_image(string path);
Mat find_all_contours(Mat& resmask, string path);
Mat find_result_contours(Mat& zero_with_contours, string path);