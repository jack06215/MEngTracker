// C++ std library
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>

// OpenCV library
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

// toolbox
#include "features2d_akaze2.hpp"
#include "fps_stats.hpp"
#include "myutils.h"

class AkazeWarp
{
// Function scope definition
public:
	AkazeWarp()

	void akaze_frontalWarp(cv::Ptr<cv::AKAZE2> detector);

private:

	void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
		std::vector<cv::Point2f>& inliers,
		float error);
	void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
		const std::vector<cv::KeyPoint>& query,
		const std::vector<std::vector<cv::DMatch> >& matches,
		std::vector<cv::Point2f>& pmatches, float nndr);

	/* New functions to be integrated */
	void cvHorCat(cv::Mat &left,
		cv::Mat &right,
		cv::Mat &res,
		int width,
		int height);

	void saveMatToCsv(cv::Mat &matrix, std::string filename);


// Parameter scope definition
public:
	fps_stats fps{ "AKAZE2" };

	cv::Ptr<cv::AKAZE2> detector;

	cv::Mat img_previous, img_current;
	std::vector<cv::KeyPoint> kp_previous, kp_current;
	cv::Mat desc_previous, desc_current;
};