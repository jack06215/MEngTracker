#pragma once
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

class Kalman
{
private:
	vector<KalmanFilter> filters;
	void initialize(int numPts);
	void initState(vector<Point2f> &pts);
	bool initialized;

public:
	Kalman(vector<Point2f> &pts);
	Kalman(int numPts);
	vector<Point2f> correct(vector<Point2f> &pts);
	vector<Point2f> predict();
	void reset();
};