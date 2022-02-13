#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#ifdef DETECTOR_EXPORTS
#    define DETECTOR_API __declspec(dllexport)
#else
#    define DETECTOR_API __declspec(dllimport)
#endif

struct openVinoRes
{
	cv::Rect coords;
	double confidence;
	int classId;
};

DETECTOR_API void* createOpenVino();
DETECTOR_API int initOpenVino(void* h_detector, std::string pathToConfig);
DETECTOR_API int calcOpenVino(void* h_detector, const cv::Mat& img, std::vector<openVinoRes>& res);