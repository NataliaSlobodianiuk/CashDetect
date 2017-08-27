#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int getCoinsSum(Mat& src);
vector<Vec3f> getCircles(Mat& src_1C, double min_radius);
int getCoinValue(Mat& img, Point center, double radius, double min_radius);
double getSaturationAvg(Mat& img, Point center);
