#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int getCoinsSum(Mat& src);
vector<Vec3f> getCircles(Mat& src_1C, double min_radius);
int getCoinValue(Mat& img, Point center, double radius, double min_radius);
double getSaturationAvg(Mat& img, Point center);

Mat getA4(Mat& src);
void fourVertecesTransform(Mat& src, vector<Point2f> verteces);
void sortVerteces(vector<Point2f>& verteces);
