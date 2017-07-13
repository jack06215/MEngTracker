#include "myutils.h"
#include <windows.h>

// Convert a vector of non-homogeneous 2D points to a vector of homogenehous 2D points.
void to_homogeneous(const std::vector< cv::Point2f >& non_homogeneous, std::vector< cv::Point3f >& homogeneous)
{
	homogeneous.resize(non_homogeneous.size());
	for (size_t i = 0; i < non_homogeneous.size(); i++)
	{
		homogeneous[i].x = non_homogeneous[i].x;
		homogeneous[i].y = non_homogeneous[i].y;
		homogeneous[i].z = 1.0;
	}
}

// Convert a vector of homogeneous 2D points to a vector of non-homogenehous 2D points.
void from_homogeneous(const std::vector< cv::Point3f >& homogeneous, std::vector< cv::Point2f >& non_homogeneous)
{
	non_homogeneous.resize(homogeneous.size());
	for (size_t i = 0; i < non_homogeneous.size(); i++)
	{
		non_homogeneous[i].x = homogeneous[i].x / homogeneous[i].z;
		non_homogeneous[i].y = homogeneous[i].y / homogeneous[i].z;
	}
}

// Transform a vector of 2D non-homogeneous points via an homography.
std::vector<cv::Point2f> transform_via_homography(const std::vector<cv::Point2f>& points, const cv::Matx33f& homography)
{
	// Convert 2D points from Cartesian coordinate to homogeneous coordinate
	std::vector<cv::Point3f> ph;
	to_homogeneous(points, ph);

	// Applied homography
	for (size_t i = 0; i < ph.size(); i++)
	{
		ph[i] = homography*ph[i];
	}

	// Convert (Normalised) the points back to Cartesian coordinate system 
	std::vector<cv::Point2f> r;
	from_homogeneous(ph, r);
	return r;
}

// Find the bounding box of a vector of 2D non-homogeneous points.
cv::Rect_<float> get_bounding_box(const std::vector<cv::Point2f>& p)
{
	cv::Rect_<float> r;
	float x_min = std::min_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.x < rhs.x; })->x;
	float x_max = std::max_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.x < rhs.x; })->x;
	float y_min = std::min_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.y < rhs.y; })->y;
	float y_max = std::max_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.y < rhs.y; })->y;
	return cv::Rect_<float>(x_min, y_min, x_max - x_min, y_max - y_min);
}

// Warp the image src into the image dst through the homography H.
void homography_warp(const cv::Mat& src, cv::Mat& H, cv::Mat& dst)
{
	// Define four corner points from the input image
	std::vector< cv::Point2f > corners;
	corners.push_back(cv::Point2f(0, 0));
	corners.push_back(cv::Point2f(src.cols, 0));
	corners.push_back(cv::Point2f(0, src.rows));
	corners.push_back(cv::Point2f(src.cols, src.rows));

	// Find the bounding box of the new corner points after applied H
	std::vector< cv::Point2f > projected_corners = transform_via_homography(corners, H);
	cv::Rect_<float> bb = get_bounding_box(projected_corners);
	//std::cout << bb << '\n';

	// Applied translation
	cv::Mat_<double> translation = (cv::Mat_<double>(3, 3) <<
		1, 0, -bb.tl().x,
		0, 1, -bb.tl().y,
		0, 0, 1);

	// Applied resultant rotation + translation warping
	cv::warpPerspective(src, dst, translation * H, bb.size());
}

void homography_warp2(const cv::Mat &image, cv::Mat &H, cv::Mat &image_out)
{

	//cv::Mat H = K_mat * R_mat * K_c * C;

	// Step 2: Calclating Resultant Translation and Scale
	std::vector<cv::Point2f> Ref_c;
	std::vector<cv::Point2f> Ref_c_out;
	Ref_c.resize(4);
	Ref_c_out.resize(4);

	Ref_c[0].x = 0;		Ref_c[0].y = 0; 									// top-left
	Ref_c[1].x = double(image.cols);	Ref_c[1].y = 0;						// top-right
	Ref_c[2].x = double(image.cols);	Ref_c[2].y = double(image.rows);	// bottom-right
	Ref_c[3].x = 0;		Ref_c[3].y = double(image.rows);					// bottom-left

	cv::perspectiveTransform(Ref_c, Ref_c_out, H);

	//Scalling:
	double scale_fac = std::abs((std::max(Ref_c_out[1].x, Ref_c_out[2].x) - std::min(Ref_c_out[0].x, Ref_c_out[3].x)) / image.cols); //Based on Length


																																	 //std::cout << scale_fac << "\t" << '\n';
																																	 //std::cout << Ref_c_out.at(0) << '\n';
																																	 // Re-scale 4 corner points by the scale_fac
	Ref_c_out[0].x = Ref_c_out[0].x / scale_fac;
	Ref_c_out[0].y = Ref_c_out[0].y / scale_fac;
	Ref_c_out[1].x = Ref_c_out[1].x / scale_fac;
	Ref_c_out[1].y = Ref_c_out[1].y / scale_fac;
	Ref_c_out[2].x = Ref_c_out[2].x / scale_fac;
	Ref_c_out[2].y = Ref_c_out[2].y / scale_fac;
	Ref_c_out[3].x = Ref_c_out[3].x / scale_fac;
	Ref_c_out[3].y = Ref_c_out[3].y / scale_fac;

	Ref_c_out[1].x = Ref_c_out[1].x - Ref_c_out[0].x;
	Ref_c_out[1].y = Ref_c_out[1].y - Ref_c_out[0].y;
	Ref_c_out[2].x = Ref_c_out[2].x - Ref_c_out[0].x;
	Ref_c_out[2].y = Ref_c_out[2].y - Ref_c_out[0].y;
	Ref_c_out[3].x = Ref_c_out[3].x - Ref_c_out[0].x;
	Ref_c_out[3].y = Ref_c_out[3].y - Ref_c_out[0].y;
	Ref_c_out[0].x = Ref_c_out[0].x - Ref_c_out[0].x;
	Ref_c_out[0].y = Ref_c_out[0].y - Ref_c_out[0].y;


	//For the translated/scalled image
	H = getPerspectiveTransform(Ref_c, Ref_c_out);

	int maxCols(0), maxRows(0), minCols(0), minRows(0);

	for (int i = 0; i<Ref_c_out.size(); i++)
	{
		if (maxRows < Ref_c_out.at(i).y)
			maxRows = Ref_c_out.at(i).y;

		else if (minRows > Ref_c_out.at(i).y)
			minRows = Ref_c_out.at(i).y;

		if (maxCols < Ref_c_out.at(i).x)
			maxCols = Ref_c_out.at(i).x;

		else if (minCols > Ref_c_out.at(i).x)
			minCols = Ref_c_out.at(i).x;
	}

	// ------------ Warp Z axix ------------------ //
	//trans4by4 = cvMakehgtform(0.0f, 0.0f, zrotate);
	//cv::Mat R_z = trans4by4(cv::Rect(0, 0, 3, 3));
	//H = H * R_z;
	homography_warp(image, H, image_out);
	//std::cout << "col ratio: " << (double)image_out.cols/image.cols << '\n';
	//std::cout << "row ratio: " << (double)image_out.rows/image.rows << '\n';

	//Ref_c_out = transform_via_homography(Ref_c, H);
}

void toGray(const cv::Mat &frame, cv::Mat &gray)
{
	if (frame.channels() > 1)
		cv::cvtColor(frame, gray, CV_BGR2GRAY);
	else
		frame.copyTo(gray);
}

void toRGB(const cv::Mat &frame, cv::Mat &rgb)
{
	if (frame.channels() == 1)
		cv::cvtColor(frame, rgb, CV_GRAY2RGB);
	else
		frame.copyTo(rgb);
}

void drawBoundingBox(cv::Mat image, std::vector<cv::Point2f> bb, cv::Scalar colour)
{
	for (unsigned i = 0; i < bb.size() - 1; i++) {
		line(image, bb[i], bb[i + 1], colour, 2);
	}
	line(image, bb[bb.size() - 1], bb[0], colour, 2);
}

void drawDot(cv::Mat image, std::vector<cv::Point2f> bb, cv::Scalar colour)
{
	for (unsigned i = 0; i < bb.size() - 1; i++) 
	{
		//line(image, bb[i], bb[i + 1], colour, 2);
		cv::circle(image, bb[i], 2, colour, 2);
	}
	cv::circle(image, bb[bb.size() - 1], 2, colour, 2);
}



std::vector<cv::Point2f> Points(std::vector<cv::KeyPoint> keypoints)
{
	std::vector<cv::Point2f> res;
	for (unsigned i = 0; i < keypoints.size(); i++) {
		res.push_back(keypoints[i].pt);
	}
	return res;
}

std::string paddingZeros(int num, int max_zeros)
{
	std::ostringstream ss;
	ss << std::setw(max_zeros) << std::setfill('0') << num;
	return ss.str();
}

std::string cv_makeFilename(std::string path_pattern, int index, cv_FileSuffix format)
{
	std::string format_1 = "";
	switch (format)
	{
	case JPEG:  format_1 += ".jpeg";
	case PNG:	format_1 += ".png";
	default:	format_1 += " ";
	}
	std::string save_index = paddingZeros(index + 1, 4);
	return (path_pattern + save_index + format_1);
}

int numDigits(int number)
{
	int digits = 0;
	if (number < 0) digits = 1; // remove this line if '-' counts as a digit
	while (number)
	{
		number /= 10;
		digits++;
	}
	return digits;
}

std::string ExePath()
{
	char buffer[MAX_PATH];
	GetModuleFileNameA(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	return std::string(buffer).substr(0, pos);
}

cv::Mat cvMakehgtform(double xrotate, double yrotate, double zrotate)
{
	// Rotation matrices around the X, Y, and Z axis
	cv::Mat RX = (cv::Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, cos(xrotate), -sin(xrotate), 0,
		0, sin(xrotate), cos(xrotate), 0,
		0, 0, 0, 1);
	cv::Mat RY = (cv::Mat_<double>(4, 4) <<
		cos(yrotate), 0, sin(yrotate), 0,
		0, 1, 0, 0,
		-sin(yrotate), 0, cos(yrotate), 0,
		0, 0, 0, 1);
	cv::Mat RZ = (cv::Mat_<double>(4, 4) <<
		cos(zrotate), -sin(zrotate), 0, 0,
		sin(zrotate), cos(zrotate), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);

	// Composed rotation matrix with (RX, RY, RZ)
	cv::Mat R = RX * RY * RZ;
	return R;
}

cv::Mat imread_limitedWidth(cv::String filename, int length_limit, int imread_flag)
{
	// Load an image
	cv::Mat image = imread(filename, imread_flag);

	if (image.empty())
	{
		std::cerr << "Cannot open image" << std::endl;
		return image;
	}

	// Resize if image length is bigger than 10000 px
	if (image.rows > 1000 | image.cols > 1000)
	{
		std::cout << "imread_resize:: Limit the image size to 1000 px in length" << std::endl;
		double fx = length_limit / static_cast<double>(image.cols);
		double newHeight = static_cast<double>(image.rows) * fx;
		double fy = newHeight / image.rows;
		cv::resize(image, image, cv::Size(0, 0), fx, fy);
	}

	return image;
}

bool isNumeric(const char* pszInput, int nNumberBase)
{
	std::string base = "0123456789ABCDEF";
	std::string input = pszInput;

	return (input.find_first_not_of(base.substr(0, nNumberBase)) == std::string::npos);
}

bool isFloat(std::string myString)
{
	std::istringstream iss(myString);
	float f;
	iss >> std::noskipws >> f;	// noskipws considers leading whitespace invalid
								// Check the entire string was consumed and if either failbit or badbit is set
	return iss.eof() && !iss.fail();
}

//     meshgridTest(cv::Range(1,3), cv::Range(10, 14), X, Y);
std::pair<cv::Mat1i, cv::Mat1i> meshgrid(const cv::Range &xgv, const cv::Range &ygv)
{
	struct meshgrid_test
	{
		static void meshgrid_test1(const cv::Mat &xgv, const cv::Mat &ygv,
			cv::Mat1i &X, cv::Mat1i &Y)
		{
			cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
			cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
		}
	};

	std::pair<cv::Mat1i, cv::Mat1i> mesh_pair;
	std::vector<int> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);
	meshgrid_test::meshgrid_test1(cv::Mat(t_x), cv::Mat(t_y), mesh_pair.first, mesh_pair.second);

	return mesh_pair;
}

inline float mod(float x, float y)
{
	float result = std::fmod(x, y);
	return result >= 0 ? result : result + y;
}

void saveMatToCsv(cv::Mat &matrix, std::string filename)
{
	std::ofstream outputFile(filename);
	outputFile << cv::format(matrix, 2) << std::endl;
	outputFile.close();
}

cv::Mat Sub2Ind(int width, int height, cv::Mat X, cv::Mat Y)
{
	/*sub2ind(size(a), rowsub, colsub)
	sub2ind(size(a), 2     , 3 ) = 6
	a = 1 2 3 ;
	4 5 6
	rowsub + colsub-1 * numberof rows in matrix*/

	std::vector<int> index;
	cv::transpose(Y, Y);
	cv::MatConstIterator_<int> iterX = X.begin<int>(), it_endX = X.end<int>();
	cv::MatConstIterator_<int> iterY = Y.begin<int>(), it_endY = Y.end<int>();
	for (int j = 0; j < X.cols; ++j, ++iterX)
	{
		//running on each col of y matrix
		for (int i = 0; i < Y.cols; ++i, ++iterY)
		{
			int rowsub = *iterY;
			int colsub = *iterX;
			int res = rowsub + ((colsub - 1)*height);
			index.push_back(res);
		}
		int x = 5;
	}
	cv::Mat M(index);
	return M;
}

cv::Rect2d selectROI(const std::string &video_name, const cv::Mat &frame)
{
	cv::namedWindow(video_name, cv::WINDOW_NORMAL);
	cv::resizeWindow(video_name, frame.cols, frame.rows);
	struct Data
	{
		cv::Point center;
		cv::Rect2d box;

		static void mouseHandler(int event, int x, int y, int flags, void *param)
		{
			Data *data = (Data*)param;
			switch (event)
			{
				// start to select the bounding box
			case cv::EVENT_LBUTTONDOWN:
				data->box = cvRect(x, y, 0, 0);
				data->center = cv::Point2f((float)x, (float)y);
				break;
				// update the selected bounding box
			case cv::EVENT_MOUSEMOVE:
				if (flags == 1)
				{
					data->box.width = 2 * (x - data->center.x);
					data->box.height = 2 * (y - data->center.y);
					data->box.x = data->center.x - data->box.width / 2.0;
					data->box.y = data->center.y - data->box.height / 2.0;
				}
				break;
				// cleaning up the selected bounding box
			case cv::EVENT_LBUTTONUP:
				if (data->box.width < 0)
				{
					data->box.x += data->box.width;
					data->box.width *= -1;
				}
				if (data->box.height < 0)
				{
					data->box.y += data->box.height;
					data->box.height *= -1;
				}
				break;
			}
		}
	} data;

	cv::setMouseCallback(video_name, Data::mouseHandler, &data);

	while (cv::waitKey(1) < 0)
	{
		cv::Mat draw = frame.clone();
		rectangle(draw, data.box, cv::Scalar(255, 0, 0), 2, 1);
		cv::imshow(video_name, draw);
	}
	cv::destroyWindow(video_name);
	return data.box;
}

cv::Mat create_homography_map(cv::Mat &img, cv::Mat &tform)
{
	// Define four corner points from the input image
	std::vector< cv::Point2f > corners;
	corners.push_back(cv::Point2f(0, 0));
	corners.push_back(cv::Point2f(img.cols, 0));
	corners.push_back(cv::Point2f(0, img.rows));
	corners.push_back(cv::Point2f(img.cols, img.rows));

	// Find the bounding box of the new corner points after applied H
	std::vector< cv::Point2f > projected_corners = transform_via_homography(corners, tform);
	cv::Rect_<float> bb = get_bounding_box(projected_corners);
	//std::cout << bb << '\n';

	// Applied translation
	cv::Mat translation = (cv::Mat_<double>(3, 3) <<
		1, 0, -bb.tl().x,
		0, 1, -bb.tl().y,
		0, 0, 1);

	cv::Mat M = translation * tform;
	return M;
}