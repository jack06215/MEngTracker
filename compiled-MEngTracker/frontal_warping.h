#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "features2d_akaze2.hpp"
#include "myutils.h"

// Akaze parameters
#define FRONTAL_DESCRIPTOR_SIZE            64     /* 64 or 256 or 486 bits; 0 means full and 486 bits in case of three channels */
#define FRONTAL_DESCRIPTOR_CH              3       /* 1 or 2 or 3; The descriptor size must be <= 162*CH */
#define FRONTAL_NUM_OCTAVES                3
#define FRONTAL_NUM_OCTAVE_SUBLAYERS       1

// dynamic threshold adjusting parameters (according to the target keypoint range)
#define FRONTAL_KPCOUNT_MIN                140
#define FRONTAL_KPCOUNT_MAX                160
#define FRONTAL_THRESHOLD_MIN              0.00001f
#define FRONTAL_THRESHOLD_MAX              0.1f


typedef std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> cvKeyPoint_pair;
typedef std::pair<cv::Mat, cv::Mat> cvMat_pair;
class mat_pair
{
public:
	void swap();

	cvMat_pair img_pair;
	cvMat_pair desc_pair;
	cvKeyPoint_pair kp_pair;
};

class frontal_warping
{
public:
	frontal_warping();
	void akaze2_frontal_warping(cvMat_pair &img_current,
		cvMat_pair &warp_current,
		cvKeyPoint_pair &kp_current,
		cvMat_pair &desc_current,
		cv::Mat &H_10);
private:
	cv::Ptr<cv::AKAZE2> detector;
};
