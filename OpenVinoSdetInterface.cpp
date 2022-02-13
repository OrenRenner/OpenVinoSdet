#define DETECTOR_EXPORTS

#include "OpenVinoSdet.h"

int initOpenVino(void* h_detector, std::string pathToConfig)
{
	if (!h_detector)return 1;

	return ((OpenVinoSdet*)h_detector)->init(pathToConfig);
};

int calcOpenVino(void* h_detector, const cv::Mat& img, std::vector<openVinoRes>& res)
{
	if (!h_detector)return 1;

	return ((OpenVinoSdet*)h_detector)->calc(img, res);
};

void* createOpenVino()
{
	return new OpenVinoSdet;
}