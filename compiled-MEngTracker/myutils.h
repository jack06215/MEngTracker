#pragma once
// C++ OpenCV 
#include <opencv2/opencv.hpp>
// C++ Math and other utility
#include <random>
#include <limits>
#include <utility>
#define NOMINMAX
#include <cmath>

//// C++11 multi-threading
//#include <functional>	// ref()
//#include <atomic>
//#include <thread>
//#include <chrono>

// C++ std IO
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <iomanip>

// OpenCV file suffix
enum cv_FileSuffix { JPEG, PNG };

// Convert a vector of non-homogeneous 2D points to a vector of homogenehous 2D points.
void to_homogeneous(const std::vector< cv::Point2f >& non_homogeneous, std::vector< cv::Point3f >& homogeneous);

// Convert a vector of homogeneous 2D points to a vector of non-homogenehous 2D points.
void from_homogeneous(const std::vector< cv::Point3f >& homogeneous, std::vector< cv::Point2f >& non_homogeneous);

// Transform a vector of 2D non-homogeneous points via an homography.
std::vector<cv::Point2f> transform_via_homography(const std::vector<cv::Point2f>& points, const cv::Matx33f& homography);

// Find the bounding box of a vector of 2D non-homogeneous points.
cv::Rect_<float> get_bounding_box(const std::vector<cv::Point2f>& p);

// Warp the "src" into the "dst" through the homography "H".
//		"homography_warp" is the original implementation
//		"homography_warp2" is the same as homography_warp, except it fixes the dimension of output image
void homography_warp2(const cv::Mat &image, cv::Mat &H, cv::Mat &image_out);
void homography_warp(const cv::Mat& src, cv::Mat& H, cv::Mat& dst);

// Convert RGB image to grayscale
void toGray(const cv::Mat &frame, cv::Mat &gray);

// Convert grayscale image to RGB
void toRGB(const cv::Mat &frame, cv::Mat &rgb);

void drawDot(cv::Mat image, std::vector<cv::Point2f> bb, cv::Scalar colour = cv::Scalar(0, 0, 255));

// Draw a bounding box on an image
void drawBoundingBox(cv::Mat image, std::vector<cv::Point2f> bb, cv::Scalar colour = cv::Scalar(0, 0, 255));

// Convert cv::KeyPoint to cv::Point2f
std::vector<cv::Point2f> Points(std::vector<cv::KeyPoint> keypoints);

// Pad 0s into a string
std::string paddingZeros(int num, int max_zeros);

// custom function for reading image sequence
std::string cv_makeFilename(std::string path_pattern, int index, cv_FileSuffix format);

// Return the number of digits of a given int
int numDigits(int number);

// (for Windows) Return the current working directory
std::string ExePath();

// Make homography transformation matrix from euler rotation.
cv::Mat cvMakehgtform(double xrotate, double yrotate, double zrotate);

// Read an image with limited length
cv::Mat imread_limitedWidth(cv::String filename, int length_limit, int imread_flag);

// Check if a string is a number
bool isNumeric(const char* pszInput, int nNumberBase);

// Check is a string is a floating number
bool isFloat(std::string myString);

// MATLAB-equivalent function
// USEAGE: meshgridTest(cv::Range(1,3), cv::Range(10, 14), X, Y);
std::pair<cv::Mat1i, cv::Mat1i> meshgrid(const cv::Range &xgv, const cv::Range &ygv);
inline float mod(float x, float y);
cv::Mat Sub2Ind(int width, int height, cv::Mat X, cv::Mat Y);


// Save Mat to CSV file
void saveMatToCsv(cv::Mat &matrix, std::string filename);

cv::Rect2d selectROI(const std::string &video_name, const cv::Mat &frame);

cv::Mat create_homography_map(cv::Mat &img, cv::Mat &tform);


