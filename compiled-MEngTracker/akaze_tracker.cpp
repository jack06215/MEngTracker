#include "akaze_tracker.h"
#include "myutils.h"
#include <opencv2/opencv.hpp>

akaze_tracker::~akaze_tracker()
{

}

void akaze_tracker::computeKeyPoints(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{
	detector_->detect(image, keypoints);
	//std::cout << keypoints.size() << '\n';

}

void akaze_tracker::computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	extractor_->compute(image, keypoints, descriptors);
}

void akaze_tracker::detectAndCompute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	detector_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

int akaze_tracker::ratioTest(std::vector<std::vector<cv::DMatch> > &matches)
{
	int removed = 0;
	// for all matches
	for (std::vector<std::vector<cv::DMatch> >::iterator
		matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator)
	{
		// if 2 NN has been identified
		if (matchIterator->size() > 1)
		{
			// check distance ratio
			if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio_)
			{
				matchIterator->clear(); // remove match
				removed++;
			}
		}
		else
		{ // does not have 2 neighbours
			matchIterator->clear(); // remove match
			removed++;
		}
	}
	return removed;
}

void akaze_tracker::symmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1, const std::vector<std::vector<cv::DMatch> >& matches2, std::vector<cv::DMatch>& symMatches)
{

	// for all matches image 1 -> image 2
	for (std::vector<std::vector<cv::DMatch> >::const_iterator
		matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1)
	{

		// ignore deleted matches
		if (matchIterator1->empty() || matchIterator1->size() < 2)
			continue;

		// for all matches image 2 -> image 1
		for (std::vector<std::vector<cv::DMatch> >::const_iterator
			matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2)
		{
			// ignore deleted matches
			if (matchIterator2->empty() || matchIterator2->size() < 2)
				continue;

			// Match symmetry test
			if ((*matchIterator1)[0].queryIdx ==
				(*matchIterator2)[0].trainIdx &&
				(*matchIterator2)[0].queryIdx ==
				(*matchIterator1)[0].trainIdx)
			{
				// add symmetrical match
				symMatches.push_back(
					cv::DMatch((*matchIterator1)[0].queryIdx,
					(*matchIterator1)[0].trainIdx,
						(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}
void akaze_tracker::tune_akazeCV_threshold(int last_nkp)
{
	if (AKAZE_KPCOUNT_MIN <= last_nkp && last_nkp <= AKAZE_KPCOUNT_MAX)
		return;

	/*
	By converting the parameters as y = log10(nkp+1), x = log10(threshold),
	a simple fitting line, y = a * x + b, can be assumed to find out
	the threshold to give the target nkp
	*/

	const double target_nkp = 0.5 * (AKAZE_KPCOUNT_MAX + AKAZE_KPCOUNT_MIN);
	const double target_y = log10(target_nkp);

	// Some negative number; closer to 0 means finer and slower to approach the target
	const double slope = -0.5;

	double x = log10(this->detector_->getHessianThreshold());
	double y = log10(last_nkp + 1.0);

	x = x + slope * (target_y - y);
	double threshold = exp(x * log(10.0));
	//std::cout << threshold << '\n';


	if (threshold > 500)
		threshold = 500; // The aperture is closed
	else
		if (threshold < 100)
			threshold = 100; // The aperture is fully open

	this->detector_->setHessianThreshold(threshold);
	//std::cout << this->detector_->getThreshold() << '\n';
}
//void akaze_tracker::tune_akazeCV_threshold(int last_nkp)
//{
//	if (AKAZE_KPCOUNT_MIN <= last_nkp && last_nkp <= AKAZE_KPCOUNT_MAX)
//		return;
//
//	/*
//	By converting the parameters as y = log10(nkp+1), x = log10(threshold),
//	a simple fitting line, y = a * x + b, can be assumed to find out
//	the threshold to give the target nkp
//	*/
//
//	const double target_nkp = 0.5 * (AKAZE_KPCOUNT_MAX + AKAZE_KPCOUNT_MIN);
//	const double target_y = log10(target_nkp);
//
//	// Some negative number; closer to 0 means finer and slower to approach the target
//	const double slope = -1.0;
//
//	double x = log10(this->detector_->getThreshold());
//	double y = log10(last_nkp + 1.0);
//
//	x = x + slope * (target_y - y);
//	double threshold = exp(x * log(10.0));
//
//	if (threshold > AKAZE_THRESHOLD_MAX)
//		threshold = AKAZE_THRESHOLD_MAX; // The aperture is closed
//	else
//		if (threshold < AKAZE_THRESHOLD_MIN)
//			threshold = AKAZE_THRESHOLD_MIN; // The aperture is fully open
//
//	this->detector_->setThreshold(threshold);
//	//std::cout << this->detector_->getThreshold() << '\n';
//}

void akaze_tracker::filterKeyPoints(const std::vector<cv::Point2f> &corners,						// warped image corner
									std::vector<cv::KeyPoint> &keypoints)							// warped keypoints
{
	std::vector<cv::KeyPoint> newKeypoints;
	newKeypoints.reserve(keypoints.size());

	for (unsigned int i = 0; i<keypoints.size(); i++)
	{
		cv::KeyPoint& k = keypoints[i];
		//reject if keypoint is too close to corner of the warped image
		if (cv::pointPolygonTest(corners, cv::Point2f(k.pt.x - HALF_PATCH_WIDTH, k.pt.y - HALF_PATCH_WIDTH), false) < 0 ||
			cv::pointPolygonTest(corners, cv::Point2f(k.pt.x - HALF_PATCH_WIDTH, k.pt.y + HALF_PATCH_WIDTH), false) < 0 ||
			cv::pointPolygonTest(corners, cv::Point2f(k.pt.x + HALF_PATCH_WIDTH, k.pt.y - HALF_PATCH_WIDTH), false) < 0 ||
			cv::pointPolygonTest(corners, cv::Point2f(k.pt.x + HALF_PATCH_WIDTH, k.pt.y + HALF_PATCH_WIDTH), false) < 0)
			continue;

		newKeypoints.push_back(k);
	}
	
	//replace old points with new points
	keypoints.clear();
	keypoints = newKeypoints;
	return;
}

void akaze_tracker::setFirstFrame(const cv::Mat& frame, std::vector<cv::Point2f> bb)
{
	cv::Mat frame_bw;
	if (frame.channels() > 1)
		toGray(frame, frame_bw);

	else
		frame.copyTo(frame_bw);

	//cv::Ptr<cv::AKAZE2> tmp_detector =  cv::AKAZE2::create(cv::AKAZE::DESCRIPTOR_MLDB, AKAZE_DESCRIPTOR_SIZE, AKAZE_DESCRIPTOR_CH, 0.01f, AKAZE_NUM_OCTAVES, AKAZE_NUM_OCTAVE_SUBLAYERS);
	cv::Ptr<cv::xfeatures2d::SURF> tmp_detector = cv::xfeatures2d::SURF::create(100.0, 4, 3, false, true);
	cv::Point *ptMask = new cv::Point[bb.size()];
	const cv::Point* ptContain = { &ptMask[0] };
	int iSize = static_cast<int>(bb.size());
	for (size_t i = 0; i<bb.size(); i++)
	{
		ptMask[i].x = static_cast<int>(bb[i].x);
		ptMask[i].y = static_cast<int>(bb[i].y);
	}
	cv::Mat matMask = cv::Mat::zeros(frame_bw.size(), CV_8UC1);
	cv::fillPoly(matMask, &ptContain, &iSize, 1, cv::Scalar::all(255));
	//tmp_detector->detectAndCompute(frame_bw, matMask, model_kpts_, model_desc_);
	tmp_detector->detectAndCompute(frame_bw, cv::noArray(), model_kpts_, model_desc_);
	//std::cout << model_kpts_.size() << '\n';

	object_bb = bb;
	delete[] ptMask;
	tmp_detector.release();
	frame_bw.release();
}

static void matches2points_nndr2(const std::vector<cv::KeyPoint>& train,
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

static void compute_inliers_ransac2(const std::vector<cv::Point2f>& matches,
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


void akaze_tracker::matching2(const std::vector<cv::KeyPoint>& train,
	const std::vector<cv::KeyPoint>& query,
	const std::vector<std::vector<cv::DMatch> >& matches,
	std::vector<cv::Point2f>& pmatches, float nndr, float error)
{
	std::vector<cv::Point2f> inliers;
	matches2points_nndr2(train, query, matches, pmatches, nndr);
	compute_inliers_ransac2(pmatches, inliers, error);
	std::vector<cv::Point2f> pts1;
	std::vector<cv::Point2f> pts2;
	//std::cout << "inlier size" << inliers.size() << '\n';
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
	vector<cv::Point2f> new_bb;
	perspectiveTransform(this->object_bb, new_bb, HG);
	//for (int i = 0; i < new_bb.size(); i++)
	//{
	//	std::cout << new_bb[i] << '\n';
	//}
	this->result.boundingBox = new_bb;
	this->result.homography = HG;

}

tracker_result akaze_tracker::process2(const cv::Mat& frame, std::vector<cv::Point2f> bb, tracker_options::detection option)
{
	using namespace tracker_options;
	tracker_result result;
	std::vector<cv::DMatch> good_matches;
	vector<cv::KeyPoint> inliers1, inliers2;
	vector<cv::DMatch> inlier_matches;
	std::vector<cv::KeyPoint> frame_keypoints;


	this->detection2(frame, bb, good_matches, frame_keypoints);


	inlier_matches.clear();
	inliers1.clear();
	inliers2.clear();
	good_matches.clear();
	frame_keypoints.clear();

	return result;
}

tracker_result akaze_tracker::process(const cv::Mat& frame, std::vector<cv::Point2f> bb, tracker_options::detection option)
{
	using namespace tracker_options;
	tracker_result result;
	std::vector<cv::DMatch> good_matches;
	vector<cv::KeyPoint> inliers1, inliers2;
	vector<cv::DMatch> inlier_matches;
	std::vector<cv::KeyPoint> frame_keypoints;

	switch (option)
	{
	case FAST:
		this->detection(frame, bb, good_matches, frame_keypoints);
		break;
	case ROBUST:
		this->robustDetection(frame, bb, good_matches, frame_keypoints);
		break;
	default:
		this->robustDetection(frame, bb, good_matches, frame_keypoints);
		break;
	}
#if 1
	this->matching(good_matches, frame_keypoints, inliers1, inliers2, inlier_matches);

	if (inlier_matches.size())
	{
		//std::cout << inlier_matches.size() << '\n';

		result.boundingBox = this->result.boundingBox;
		result.homography = this->result.homography;
		result.inliers1 = inliers1;
		result.inliers2 = inliers2;
		result.inlier_matches = inlier_matches;
	}
#endif
	inlier_matches.clear();
	inliers1.clear();
	inliers2.clear();
	good_matches.clear();
	frame_keypoints.clear();

	return result;
}

void akaze_tracker::matching(std::vector<cv::DMatch>& good_matches, std::vector<cv::KeyPoint>& frame_keypoints,
	std::vector<cv::KeyPoint>& inliers1, std::vector<cv::KeyPoint>& inliers2, std::vector<cv::DMatch>& inlier_matches)
{
	vector<cv::KeyPoint> matched1, matched2;
	cv::Mat inlier_mask, Htform;
	for (std::vector<cv::DMatch>::iterator matchIter = good_matches.begin(); matchIter != good_matches.end(); ++matchIter)
	{
		matched1.push_back(model_kpts_[matchIter->trainIdx]);
		matched2.push_back(frame_keypoints[matchIter->queryIdx]);
	}
	if (matched1.size() >= 4)
	{
		Htform = findHomography(Points(matched1), Points(matched2), cv::RANSAC, 2.5f, inlier_mask);
	}
	for (unsigned j = 0; j < matched1.size(); j++)
	{
		if (!inlier_mask.empty())
		{
			if (inlier_mask.at<uchar>(j))
			{
				int new_j = static_cast<int>(inliers1.size());
				inliers1.push_back(matched1[j]);
				inliers2.push_back(matched2[j]);
				inlier_matches.push_back(cv::DMatch(new_j, new_j, 0));
			}
		}
		else
		{
			return;
		}
		vector<cv::Point2f> new_bb;
		perspectiveTransform(object_bb, new_bb, Htform);
		this->result.boundingBox = new_bb;
		this->result.homography = Htform;
	}
}



void akaze_tracker::robustDetection(const cv::Mat& frame, const std::vector<cv::Point2f>& frame_corners, std::vector<cv::DMatch>& good_matches, std::vector<cv::KeyPoint>& keypoints_frame)
{
	good_matches.clear();
	static int last_nkp = 0;
	if (this->dynamic_threshold)
	{
		tune_akazeCV_threshold(last_nkp);
	}
	cv::Mat frame_bw;
	if (frame.channels() > 1)
		toGray(frame, frame_bw);

	else
		frame.copyTo(frame_bw);
	
	// 1a. Detection of the AKAZE features
	this->computeKeyPoints(frame_bw, keypoints_frame);
	filterKeyPoints(frame_corners, keypoints_frame);
	last_nkp = keypoints_frame.size();
	// 1b. Extraction of the AKAZE descriptors
	cv::Mat descriptors_frame;
	this->computeDescriptors(frame_bw, keypoints_frame, descriptors_frame);

	// 2. Match the two image descriptors
	std::vector<std::vector<cv::DMatch> > matches12, matches21;

	// 2a. From image 1 to image 2
	matcher_->knnMatch(descriptors_frame, model_desc_, matches12, 2); // return 2 nearest neighbours

	// 2b. From image 2 to image 1
	matcher_->knnMatch(model_desc_, descriptors_frame, matches21, 2); // return 2 nearest neighbours

	// 3. Remove matches for which NN ratio is > than threshold
	// clean image 1 -> image 2 matches
	ratioTest(matches12);
	// clean image 2 -> image 1 matches
	ratioTest(matches21);

	// 4. Remove non-symmetrical matches
	symmetryTest(matches12, matches21, good_matches);
	frame_bw.release();

}

void akaze_tracker::detection2(const cv::Mat& frame, const std::vector<cv::Point2f>& frame_corners, std::vector<cv::DMatch>& good_matches, std::vector<cv::KeyPoint>& keypoints_frame)
{
	good_matches.clear();
	static int last_nkp = 0;
	if (this->dynamic_threshold)
	{
		tune_akazeCV_threshold(last_nkp);
	}
	cv::Mat frame_bw;
	if (frame.channels() > 1)
		toGray(frame, frame_bw);

	else
		frame.copyTo(frame_bw);

	// 1a. Detection of the AKAZE features
	this->computeKeyPoints(frame_bw, keypoints_frame);
	filterKeyPoints(frame_corners, keypoints_frame);
	last_nkp = keypoints_frame.size();
	// 1b. Extraction of the AKAZE descriptors
	cv::Mat descriptors_frame;
	std::vector<cv::Point2f> pmatches;
	this->computeDescriptors(frame_bw, keypoints_frame, descriptors_frame);

	// 2. Match the two image descriptors
	std::vector<std::vector<cv::DMatch> > matches;
	matcher_->knnMatch(descriptors_frame, model_desc_, matches, 2);

	matching2(this->model_kpts_, keypoints_frame, matches, pmatches, 0.8f, 2.5f);
	frame_bw.release();
}

void akaze_tracker::detection(const cv::Mat& frame, const std::vector<cv::Point2f>& frame_corners, std::vector<cv::DMatch>& good_matches, std::vector<cv::KeyPoint>& keypoints_frame)
{
	good_matches.clear();
	static int last_nkp = 0;
	if (this->dynamic_threshold)
	{
		tune_akazeCV_threshold(last_nkp);
	}
	cv::Mat frame_bw;
	if (frame.channels() > 1)
		toGray(frame, frame_bw);

	else
		frame.copyTo(frame_bw);
	
	cv::Mat descriptors_frame;
	//std::cout << this->detector_->getHessianThreshold() << '\n';
	// 1a. Detection of the AKAZE features
	this->computeKeyPoints(frame_bw, keypoints_frame);
	
	filterKeyPoints(frame_corners, keypoints_frame);
	last_nkp = (int)keypoints_frame.size();
	// 1b. Extraction of the AKAZE descriptors
	this->computeDescriptors(frame_bw, keypoints_frame, descriptors_frame);

	//this->detectAndCompute(frame_bw, keypoints_frame, descriptors_frame);
	//std::cout << descriptors_frame.size() << '\n';
#if 1
	// 2. Match the two image descriptors
	std::vector<std::vector<cv::DMatch> > matches;
	matcher_->knnMatch(descriptors_frame, model_desc_, matches, 2);
	//std::cout << "yyyyyyyyyyyy" << '\n';
	// 3. Remove matches for which NN ratio is > than threshold
	ratioTest(matches);

	// 4. Fill good matches container
	for (std::vector<std::vector<cv::DMatch> >::iterator
		matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator)
	{
		if (!matchIterator->empty()) 
			good_matches.push_back((*matchIterator)[0]);
	}
#endif
	frame_bw.release();
}
