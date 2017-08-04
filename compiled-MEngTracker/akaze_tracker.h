#pragma once
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "features2d_akaze2.hpp"

// Akaze parameters
#define AKAZE_DESCRIPTOR_SIZE            0     /* 64 or 256 or 486 bits; 0 means full and 486 bits in case of three channels */
#define AKAZE_DESCRIPTOR_CH              3      /* 1 or 2 or 3; The descriptor size must be <= 162*CH */
#define AKAZE_NUM_OCTAVES                4
#define AKAZE_NUM_OCTAVE_SUBLAYERS       4

// dynamic threshold adjusting parameters (according to the target keypoint range)
#define AKAZE_KPCOUNT_MIN                140
#define AKAZE_KPCOUNT_MAX                160
#define AKAZE_THRESHOLD_MIN              0.00001f
#define AKAZE_THRESHOLD_MAX              0.1f

#define HALF_PATCH_WIDTH				10	

namespace tracker_options
{
	enum detection { FAST, ROBUST };
}

typedef struct
{
	// Detection result
	std::vector<cv::Point2f> boundingBox;	// Bounding box of tracking result in current frame
	cv::Mat homography;						// Transformation from reference to current frame
	
	// For drawMatches() function
	std::vector<cv::KeyPoint> inliers1;
	std::vector<cv::KeyPoint> inliers2;
	std::vector<cv::DMatch> inlier_matches;
} tracker_result;

class akaze_tracker
{
public:
	akaze_tracker() : ratio_(0.8), dynamic_threshold(true)
	{
		//detector_ = cv::AKAZE2::create(cv::AKAZE::DESCRIPTOR_MLDB, AKAZE_DESCRIPTOR_SIZE,AKAZE_DESCRIPTOR_CH,0.01f, AKAZE_NUM_OCTAVES, AKAZE_NUM_OCTAVE_SUBLAYERS);
		//extractor_ = cv::AKAZE2::create(cv::AKAZE::DESCRIPTOR_MLDB, AKAZE_DESCRIPTOR_SIZE, AKAZE_DESCRIPTOR_CH, 0.01f, AKAZE_NUM_OCTAVES, AKAZE_NUM_OCTAVE_SUBLAYERS);
		detector_ = cv::FastFeatureDetector::create();
		extractor_ = cv::BRISK::create();
		matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");
	}
	virtual ~akaze_tracker();
	void setFirstFrame(cv::Mat& frame, std::vector<cv::Point2f> bb);
	void matching(std::vector<cv::DMatch>& good_matches, std::vector<cv::KeyPoint>& frame_keypoints,
		std::vector<cv::KeyPoint>& inliers1, std::vector<cv::KeyPoint>& inliers2, std::vector<cv::DMatch>& inlier_matches);
	void detection(const cv::Mat& frame, const std::vector<cv::Point2f>& frame_corners, std::vector<cv::DMatch>& good_matches, std::vector<cv::KeyPoint>& keypoints_frame);
	void robustDetection(const cv::Mat& frame, const std::vector<cv::Point2f>& frame_corners, std::vector<cv::DMatch>& good_matches, std::vector<cv::KeyPoint>& keypoints_frame);	
	tracker_result process(const cv::Mat& frame, std::vector<cv::Point2f> bb, tracker_options::detection option);
	tracker_result result;
protected:
	void detectAndCompute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
	void computeKeyPoints(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
	void computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
	int ratioTest(std::vector<std::vector<cv::DMatch> > &matches);
	void symmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1, const std::vector<std::vector<cv::DMatch> >& matches2, std::vector<cv::DMatch>& symMatches);
	void tune_akazeCV_threshold(int last_nkp);
	void filterKeyPoints(const std::vector<cv::Point2f> &corners, std::vector<cv::KeyPoint> &keypoints);
private:
	cv::Ptr<cv::FastFeatureDetector> detector_;
	cv::Ptr<cv::BRISK> extractor_;
	cv::Ptr<cv::DescriptorMatcher> matcher_;
	std::vector<cv::KeyPoint> model_kpts_;
	cv::Mat model_desc_;
	std::vector<cv::Point2f> object_bb;
	float ratio_;
	float akaze_threshold;
	bool dynamic_threshold;
};