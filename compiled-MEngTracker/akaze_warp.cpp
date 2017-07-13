#include "akaze_warp.h"


void AkazeWarp::akaze_frontalWarp(cv::Ptr<cv::AKAZE2> detector)
{
	fps.tick();
	std::string path = ".\\sampleA\\result_";
	std::string format_1 = ".png";
	static int last_nkp = 0;
	std::string save_index = paddingZeros(i + 1, 4);
	cv::Mat img_current = cv::imread(path + save_index + format_1, 0);
	if (img_current.empty())
	{
		std::cerr << "Cannot open image frame, exiting the progam" << std::endl;
		return;
	}

	// Create a Brute-force matcher and the related constructs
	auto matcher = cv::BFMatcher{ cv::NORM_HAMMING, /* crossCheck */ true };
	std::vector<cv::DMatch> outliers1, outliers2, outliers3;
	std::vector<cv::Point2f> matches;
	std::vector<cv::Point2f> inliers;

	// Keypoint detection and build descriptor
	tune_akaze_threshold(*detector, last_nkp);
	kp_current.clear();
	desc_current.release();
	detector->detectAndCompute(img_current, cv::noArray(), kp_current, desc_current);
	matches.clear();
	last_nkp = (int)kp_current.size();

	// descriptor matching
	cv::Ptr<cv::DescriptorMatcher> matcher_l2 = cv::DescriptorMatcher::create("BruteForce-Hamming");
	std::vector<std::vector<cv::DMatch> > dmatches;

	// continue only if there are keypoints
	if (last_nkp > 0 && kp_previous.size() > 0)
	{
		// matching
		matcher_l2->knnMatch(desc_previous, desc_current, dmatches, 2);
		matches2points_nndr(kp_previous, kp_current, dmatches, matches, 0.80f);

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
		cv::Mat H_20 = H_10 * HG.inv();
		H_10 = H_20;

		cv::Mat frontal_est;
		homography_warp(img_current, H_20, frontal_est);

		// show the result
		cv::Mat res;
		cvHorCat(img_current, frontal_est, res, 640, 480);
		cv::imshow(WIN_TITLE_OUTPUT, res);

		// save Mat to csv
#if 0
		path = ".\\saveMat\\";
		std::string saveMat = "H_" + save_index + ".jpg";
		cv::imwrite(path + saveMat, frontal_est);
		saveMat = "H_" + save_index + ".csv";
		saveMatToCsv(H_10, path + saveMat);
#endif
	}

	// memory handling and freeing
	img_previous = img_current.clone();
	std::swap(kp_current, kp_previous);
	desc_previous = desc_current.clone();
}

void AkazeWarp::compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
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

void AkazeWarp::matches2points_nndr(const std::vector<cv::KeyPoint>& train,
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

/* New functions to be integrated */
void AkazeWarp::cvHorCat(cv::Mat &left,
	cv::Mat &right,
	cv::Mat &res,
	int width,
	int height)
{
	cv::resize(right, right, cv::Size(width, height));
	cv::resize(left, left, cv::Size(width, height));
	cv::hconcat(left, right, res);
}

void AkazeWarp::saveMatToCsv(cv::Mat &matrix, std::string filename)
{
	std::ofstream outputFile(filename);
	outputFile << cv::format(matrix, 2) << std::endl;
	outputFile.close();
}