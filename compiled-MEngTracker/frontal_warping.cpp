#include "frontal_warping.h"
#include "features2d_akaze2.hpp"


void mat_pair::swap()
{
	std::swap(kp_pair.first, kp_pair.second);
	
	desc_pair.first.copyTo(desc_pair.second);
	desc_pair.first.release();

	img_pair.first.copyTo(img_pair.second);
	img_pair.first.release();
}

static void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
	const std::vector<cv::KeyPoint>& query,
	const std::vector<std::vector<cv::DMatch> >& matches,
	std::vector<cv::Point2f>& pmatches, float nndr)

{

	float dist1 = 0.0, dist2 = 0.0;
	for (size_t i = 0; i < matches.size(); i++)
	{
		cv::DMatch dmatch = matches[i][0];
		dist1 = matches[i][0].distance;
		dist2 = matches[i][1].distance;

		if (dist1 < nndr*dist2) {
			pmatches.push_back(train[dmatch.queryIdx].pt);
			pmatches.push_back(query[dmatch.trainIdx].pt);
		}
	}
}

static void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
	std::vector<cv::Point2f>& inliers,
	float error)
{
	std::vector<cv::Point2f> points1, points2;
	cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);
	int npoints = matches.size() / 2;
	cv::Mat status = cv::Mat::zeros(npoints, 1, CV_8UC1);

	for (size_t i = 0; i < matches.size(); i += 2)
	{
		points1.push_back(matches[i]);
		points2.push_back(matches[i + 1]);
	}

	if (npoints > 8)
	{
		H = cv::findHomography(points1, points2, cv::RANSAC, error, status);

		for (int i = 0; i < npoints; i++)
		{
			if (status.at<unsigned char>(i) == 1)
			{
				inliers.push_back(points1[i]);
				inliers.push_back(points2[i]);
			}
		}
	}
}

frontal_warping::frontal_warping()
{
	this->detector = cv::AKAZE2::create(cv::AKAZE::DESCRIPTOR_MLDB, FRONTAL_DESCRIPTOR_SIZE, FRONTAL_DESCRIPTOR_CH, FRONTAL_THRESHOLD_MAX, FRONTAL_NUM_OCTAVES, FRONTAL_NUM_OCTAVE_SUBLAYERS);
}

void tune_akaze_threshold(cv::AKAZE2 & akaze_,
	int last_nkp)
{
	if (FRONTAL_KPCOUNT_MIN <= last_nkp && last_nkp <= FRONTAL_KPCOUNT_MAX)
		return;

	/*
	By converting the parameters as y = log10(nkp+1), x = log10(threshold),
	a simple fitting line, y = a * x + b, can be assumed to find out
	the threshold to give the target nkp
	*/

	const double target_nkp = 0.5 * (FRONTAL_KPCOUNT_MAX + FRONTAL_KPCOUNT_MIN);
	const double target_y = log10(target_nkp);

	// Some negative number; closer to 0 means finer and slower to approach the target
	const double slope = -0.5;

	double x = log10(akaze_.getThreshold());
	double y = log10(last_nkp + 1.0);

	x = x + slope * (target_y - y);

	double threshold = exp(x * log(10.0));

	if (threshold > FRONTAL_THRESHOLD_MAX)
		threshold = FRONTAL_THRESHOLD_MAX; // The aperture is closed
	else
		if (threshold < FRONTAL_THRESHOLD_MIN)
			threshold = FRONTAL_THRESHOLD_MIN; // The aperture is fully open

													  //std::cout << s << " " << last_nkp << "\tdelta:" << (target_y - y) << ": " << threshold << std::endl;
													  //std::cout << s;
	akaze_.setThreshold(threshold);
}

void frontal_warping::akaze2_frontal_warping(cvMat_pair &img_current,
	cvMat_pair &warp_current,
	//cv::Ptr<cv::AKAZE2> detector,
	cvKeyPoint_pair &kp_current,
	cvMat_pair &desc_current,
	cv::Mat &H_10)
{
	static int last_nkp = 0;

	// Create a Brute-force matcher and the related constructs
	//auto matcher = cv::BFMatcher{ cv::NORM_HAMMING, /* crossCheck */ true };
	std::vector<cv::Point2f> matches;
	std::vector<cv::Point2f> inliers;

	// AKAZE Keypoint detection and build descriptor
	tune_akaze_threshold(*this->detector, last_nkp);
	kp_current.first.clear();
	desc_current.first.release();
	cv::Mat tmp_imgCurrent;
	toGray(img_current.first, tmp_imgCurrent);
	this->detector->detectAndCompute(tmp_imgCurrent, cv::noArray(), kp_current.first, desc_current.first);
	matches.clear();

	// update the size of detected for detector tunning for the next run
	last_nkp = (int)kp_current.first.size();

	// descriptor matching
	cv::Ptr<cv::DescriptorMatcher> matcher_l2 = cv::DescriptorMatcher::create("BruteForce-Hamming");
	std::vector<std::vector<cv::DMatch> > dmatches;

	// continue only if there are keypoints
	if (last_nkp > 0 && kp_current.second.size() > 0)
	{
		// matching
		matcher_l2->knnMatch(desc_current.second, desc_current.first, dmatches, 2);
		matches2points_nndr(kp_current.second, kp_current.first, dmatches, matches, 0.80f);

		//std::cout << kp_current.first.size() << "\t" << kp_current.second.size() << '\n';

		compute_inliers_ransac(matches, inliers, 2.5f);
		std::vector<cv::Point2f> pts1;
		std::vector<cv::Point2f> pts2;
		for (size_t i = 0; i < inliers.size(); i += 2)
		{
			cv::Point2f pts_tmp;
			pts_tmp.x = (int)(inliers[i].x + .5);
			pts_tmp.y = (int)(inliers[i].y + .5);
			pts1.push_back(pts_tmp);

			pts_tmp.x = (int)(inliers[i + 1].x + .5);
			pts_tmp.y = (int)(inliers[i + 1].y + .5);
			pts2.push_back(pts_tmp);
		}

		// find correspondance between consecutive frames, and deduce the pose homography of the current frame
		cv::Mat HG = findHomography(pts1, pts2);

		H_10 = H_10 * HG.inv(); // (current F to frontal) * (inverse of current and its consecutive one)

		cv::Mat frontal_est;
		//rotate_camera(img_current, warp_current, H_10);
		homography_warp2(img_current.first, H_10, warp_current.first);
	}
	tmp_imgCurrent.release();
}