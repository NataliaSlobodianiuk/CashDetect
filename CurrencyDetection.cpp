// include header file with declarations
#include"CurrencyDetection.h"


float angleBetween(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3)
{
	// Compute lengths
	double len1 = sqrt((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y));
	double len2 = sqrt((p3.x - p2.x)*(p3.x - p2.x) + (p3.y - p2.y)*(p3.y - p2.y));

	// Compute dots
	double dot = sqrt(fabs((p2.x - p1.x)*(p3.x - p2.x)) + fabs((p2.y - p1.y)*(p3.y - p2.y)));

	// Compute cos
	double a = dot / (len1 * len2);
	// Return
	return a;
}


bool verifyRectangle(const cv::Mat& image, const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4)
{
	// Verify points
	if (p1.x >= 0 && p1.x < image.cols && p1.y >= 0 && p1.y < image.rows && p2.x >= 0 && p2.x < image.cols && p2.y >= 0 && p2.y < image.cols
		&& p3.x >= 0 && p3.x < image.cols && p3.y >= 0 && p3.y < image.cols && p4.x >= 0 && p4.x < image.cols && p4.y >= 0 && p4.y < image.cols) {
		if ((fabs(angleBetween(p1, p2, p3)) < 0.01) && (fabs(angleBetween(p2, p3, p4)) < 0.01) && (fabs(angleBetween(p3, p4, p1)) < 0.01) && (fabs(angleBetween(p4, p1, p2)) < 0.01))
		{
			std::vector<cv::Point> cont = { p1, p2, p3, p4 };
			if (cv::contourArea(cont) > 0.01 * image.cols * image.rows)
			{
				// check if diagonal is bigger than sides
				double a = sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
				double b = sqrt((p3.x - p2.x) * (p3.x - p2.x) + (p3.y - p2.y) * (p3.y - p2.y));
				double c = sqrt((p3.x - p1.x) * (p3.x - p1.x) + (p3.y - p1.y) * (p3.y - p1.y));
				if (c > b && c > a) {
					// Check if contour is convex
					if (cv::isContourConvex(cont))
					{
						return true;
					}
					else {
						return false;
					}

				}
				else {
					return false;
				}

			}
			else {
				return false;
			}
		}
		else {
			return false;
		}
	}
	else {
		return false;
	}
}


std::vector<cv::Mat> getTemplates()
{
	std::vector<cv::Mat> templates;
	// open filename with templates filenames
	std::ifstream fin(TEMPATES_FILENAME);
	if (fin.is_open())
	{
		std::string line;
		while (!fin.eof())
		{
			// Get filename without extension
			getline(fin, line);
			// load image
			cv::Mat temp = cv::imread(TEMPLATES_DIR + line + EXTENSION);
			templates.push_back(temp);
		}
	}
	return templates;
}


cv::Mat rotateAndResize(const cv::Mat& image)
{
	// Image for output
	cv::Mat output = image.clone();
	if (image.cols > 1200 || image.rows > 1200)
	{
		// rotate 
		if (image.rows > image.cols)
		{
			cv::rotate(output, output, cv::ROTATE_90_CLOCKWISE);
		}
		cv::resize(output, output, cv::Size(1200, 800));
	}
	return output;
}


double getCurrencySum(const cv::Mat& image, cv::Mat& drawing)
{
	// variable for counting 
	double sum = 0;

	// variable for number of elements detected in image
	int elementsDetected = 0;

	// Detect keypoints with SIFT
	cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();

	// get templates
	std::vector<cv::Mat> templates = getTemplates();

	// Get keypoints and descriptors for templates
	std::vector<std::vector<cv::KeyPoint>> templatesKeypoints(templates.size());
	std::vector<cv::Mat> templatesDescriptors(templates.size());

	for (int i = 0; i < templates.size(); ++i)
	{
		detector->detectAndCompute(templates[i], cv::Mat(), templatesKeypoints[i], templatesDescriptors[i]);
	}

	// Copy image to save it
	cv::Mat temp = image.clone();

	// Mask for image
	cv::Mat mask(image.size(), CV_8U, cv::Scalar::all(255));

	int bfNormType = cv::NORM_L2;
	cv::BFMatcher matcher(bfNormType);

	// Look for something in hte image while there are enough keypoints
	// Go throw all templates
	for (int k = 0; k < templates.size(); ++k)
	{
		if (!templates[k].empty()) {

			// variable for checking if sth was detected
			bool detecting = true;
			while (detecting && elementsDetected <= 10) {
				// make detecting false
				detecting = false;
				// image for masking
				cv::Mat imageForDetection(temp.size(), temp.type(), cv::Scalar::all(0));
				// mask it
				temp.copyTo(imageForDetection, mask);

				// Detect keypoints and compute descriptors
				std::vector<cv::KeyPoint> keypoint;
				cv::Mat descriptors;
				detector->detectAndCompute(imageForDetection, cv::Mat(), keypoint, descriptors);

				double max_dist = 0; double min_dist = 100;

				// Check if there are keypoints and descriptors
				if (!templatesDescriptors[k].empty() && !templatesKeypoints[k].empty() && !descriptors.empty()) {
					// Get matches
					std::vector< cv::DMatch > matches;
					matcher.match(templatesDescriptors[k], descriptors, matches);

					// Check if there are some matches
					if (matches.size() > 0) {
						for (int j = 0; j < templatesDescriptors[k].rows; ++j)
						{
							double dist = matches[j].distance;
							if (dist < min_dist) min_dist = dist;
							if (dist > max_dist) max_dist = dist;
						}

						// Get only "good" matches (i.e. whose distance is less than 3*min_dist )
						std::vector< cv::DMatch > good_matches;
						for (int m = 0; m < templatesDescriptors[k].rows; ++m)
						{
							if (matches[m].distance <= 3 * min_dist)
							{
								good_matches.push_back(matches[m]);
							}
						}

						// Check if there is good matches
						if (good_matches.size() > 0) {
							//-- Localize the object
							std::vector<cv::Point2f> obj;
							std::vector<cv::Point2f> scene;
							for (size_t l = 0; l < good_matches.size(); ++l)
							{
								//-- Get the keypoints from the good matches
								obj.push_back(templatesKeypoints[k][good_matches[l].queryIdx].pt);
								scene.push_back(keypoint[good_matches[l].trainIdx].pt);
							}

							// Check if obj and scene is not empty
							if (obj.size() >= 4 && scene.size() >= 4) {
								cv::Mat H = Transformations::findHomography(obj, scene, cv::RANSAC);
								if (!H.empty()) {
									//-- Get the corners from the image_1 ( the object to be "detected" )
									std::vector<cv::Point2f> obj_corners(4);
									obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(templates[k].cols, 0);
									obj_corners[2] = cvPoint(templates[k].cols, templates[k].rows); obj_corners[3] = cvPoint(0, templates[k].rows);
									std::vector<cv::Point2f> scene_corners(4);
									perspectiveTransform(obj_corners, scene_corners, H);
									if (verifyRectangle(temp, scene_corners[0], scene_corners[1], scene_corners[2], scene_corners[3])) {
										// make detecting true again as we detected sth
										detecting = true;

										// add element detected
										++elementsDetected;

										//-- Draw lines between the corners 
										line(temp, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
										line(temp, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
										line(temp, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
										line(temp, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);

										//	Draw contour at mask
										std::vector<cv::Point> cont = { scene_corners[0], scene_corners[1], scene_corners[2], scene_corners[3] };
										std::vector<std::vector<cv::Point >> contours;
										contours.push_back(cont);
										cv::drawContours(mask, contours, -1, cv::Scalar::all(0), -1);

										// Add sum and put text
										if (k == 0 || k == 1)
										{
											cv::putText(temp, std::to_string(1), (scene_corners[0] + scene_corners[2]) / 2, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
											sum += 1;
										}
										if (k == 2 || k == 3)
										{
											cv::putText(temp, std::to_string(2), (scene_corners[0] + scene_corners[2]) / 2, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
											sum += 2;
										}
										if (k == 4 || k == 5)
										{
											cv::putText(temp, std::to_string(5), (scene_corners[0] + scene_corners[2]) / 2, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
											sum += 5;
										}
										if (k == 6 || k == 7)
										{
											cv::putText(temp, std::to_string(10), (scene_corners[0] + scene_corners[2]) / 2, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
											sum += 10;
										}
										if (k == 8 || k == 9)
										{
											cv::putText(temp, std::to_string(20), (scene_corners[0] + scene_corners[2]) / 2, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
											sum += 20;
										}
										if (k == 10 || k == 11)
										{
											cv::putText(temp, std::to_string(50), (scene_corners[0] + scene_corners[2]) / 2, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
											sum += 50;
										}
										if (k == 12 || k == 13)
										{
											cv::putText(temp, std::to_string(100), (scene_corners[0] + scene_corners[2]) / 2, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
											sum += 100;
										}
										if (k == 14 || k == 15)
										{
											cv::putText(temp, std::to_string(200), (scene_corners[0] + scene_corners[2]) / 2, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
											sum += 200;
										}
										if (k == 16 || k == 17)
										{
											cv::putText(temp, std::to_string(500), (scene_corners[0] + scene_corners[2]) / 2, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
											sum += 500;
										}
									}// chaeck if rectangle vefiried								
								}// check if H is empty
							}// check if obj and scene is not empty
						}// chaeck if there is good points
					} // check if there are some matches
				}// check if there are keypoints and descriptors
			}// check if template image is empty	
		}// check for keypoints
	}

	// Create image for drawing
	drawing = cv::Mat(temp.rows + 40, temp.cols, temp.type(), cv::Scalar::all(255));

	// Copy image to drawing
	temp.copyTo(drawing(cv::Rect(cv::Point(0, 40), cv::Point(drawing.cols, drawing.rows))));

	// Draw total sum at image
	std::string total = "Total sum: ";
	cv::putText(drawing, total + std::to_string(sum), (cv::Point(0, 0) + cv::Point(image.cols - 1, 105)) / 3, cv::FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, cv::Scalar(255, 0, 0), 2);

	return sum;
}
