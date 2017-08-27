#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

Mat getA4(Mat& src);
void fourVertecesTransform(Mat& src, vector<Point2f> verteces);
void sortVerteces(vector<Point2f>& verteces);