#include "OpenVinoSdetInterface.h"

class OpenVinoSdet
{
public:
	int init(std::string path);
	int calc(const cv::Mat& img, std::vector<openVinoRes>& res);
private:
	struct Params {
		float anchors[18] = { 12.0, 16.0, 19.0, 36.0, 40.0, 28.0, 36.0, 75.0, 76.0, 55.0, 72.0, 146.0, 142.0, 110.0, 192.0, 243.0, 459.0, 401.0 };
		int axis = 1;
		int coords = 4;
		int classes = 6;
		int end_axis = 3;
		int num = 9;
		bool do_softmax = false;
		int mask[3] = { 0, 0, 0 };
	};


	class YoloParams {
	public:
		YoloParams(Params& param, int side);
		Params param;
		int side;
		bool isYoloV3;
		std::vector<int> anchors;
	};

	struct Objects {
		int xmin;
		int ymin;
		int xmax;
		int ymax;
		int class_id;
		float confidence;
	};

	cv::dnn::Net net;

	const int NUM_CLASSES = 6;
	const float CONF_THRESH = 0.25;
	const float NMS_THRESH = 0.5;
	Params output_layers_params[3];

	float intersection_over_union(const Objects& box_1, const Objects& box_2);
};
