#include "OpenVinoSdet.h"

OpenVinoSdet::YoloParams::YoloParams(Params& param, int side) {
	if (param.anchors && param.classes && param.coords) {
		this->param = param;
		this->side = side;
	}
	else {
		this->side = 0;
	}

	this->isYoloV3 = false;
	if (param.mask) {
		this->param.num = 3;
		this->isYoloV3 = true;

		for (int i = 0; i < 3; i++) {
			this->anchors.push_back(param.anchors[param.mask[i] * 2]);
			this->anchors.push_back(param.anchors[param.mask[i] * 2 + 1]);
		}
	}
}


int OpenVinoSdet::init(std::string path)
{
	auto backend = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE;
	auto target = cv::dnn::DNN_TARGET_OPENCL;

	std::vector<cv::String> fn;
	std::string tmp_m = path;
	if (tmp_m.back() != '\\') tmp_m += '\\';
	cv::glob(tmp_m + "*.xml", fn, false);

	if (fn.size() == 0)	return 1;
	std::string xml_model = fn[0];

	fn.clear();
	cv::glob(tmp_m + "*.bin", fn, false);
	if (fn.size() == 0)	return 1;
	std::string bin_model = fn[0];

	fn.clear();
	cv::glob(tmp_m + "*.txt", fn, false);
	std::string txt_config = "";
	if (fn.size() != 0) {
		txt_config = fn[0];
	}

	try {
		this->net = cv::dnn::readNet(xml_model, bin_model);

		this->output_layers_params[0].mask[0] = 0;
		this->output_layers_params[0].mask[1] = 1;
		this->output_layers_params[0].mask[2] = 2;
		this->output_layers_params[1].mask[0] = 6;
		this->output_layers_params[1].mask[1] = 7;
		this->output_layers_params[1].mask[2] = 8;
		this->output_layers_params[2].mask[0] = 3;
		this->output_layers_params[2].mask[1] = 4;
		this->output_layers_params[2].mask[2] = 5;
		
		if (txt_config == "") {
			// Set OpenVINO support
			this->net.setPreferableBackend(backend);
			this->net.setPreferableTarget(target);

			std::printf("Target and Backend not loaded! Set default DNN_BACKEND_INFERENCE_ENGINE and DNN_TARGET_OPENCL\n");
		}
		else {
			std::ifstream infile(txt_config.c_str());
			std::string line;
			std::vector<std::string> confs;
			while (std::getline(infile, line)) {
				confs.push_back(line);
			}

			if (confs.size() < 2) {
				this->net.setPreferableBackend(backend);
				this->net.setPreferableTarget(target);

				std::printf("Target and Backend not loaded! Set default DNN_BACKEND_INFERENCE_ENGINE and DNN_TARGET_OPENCL\n");
			} else {
				if (confs[0] == "DNN_BACKEND_DEFAULT") this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
				else if (confs[0] == "DNN_BACKEND_INFERENCE_ENGINE") this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
				else if (confs[0] == "DNN_BACKEND_OPENCV") this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
				else this->net.setPreferableBackend(backend);

				if (confs[1] == "DNN_TARGET_CPU") this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
				else if (confs[1] == "DNN_TARGET_OPENCL") this->net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
				else if (confs[1] == "DNN_TARGET_OPENCL_FP16") this->net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
				else if (confs[1] == "DNN_TARGET_MYRIAD") this->net.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);
				else this->net.setPreferableTarget(target);

				std::printf("Target and Backend loaded! They are %s and %s \n", confs[0].c_str(), confs[1].c_str());
			}
			confs.clear();
		}
		return 0;
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
		return 1;
	}
	return 1;
}

int OpenVinoSdet::calc(const cv::Mat& img, std::vector<openVinoRes>& res)
{
	// Read image
	if (img.empty()) return 1;

	// Image preprocess
	int SPATIAL_SIZE = 608;
	cv::Mat input_blob;
	input_blob = cv::dnn::blobFromImage(img, 1.0, cv::Size(SPATIAL_SIZE, SPATIAL_SIZE), cv::Scalar(), false, false, CV_8U);
	std::vector<cv::String> layer_names = this->net.getUnconnectedOutLayersNames();

	if (layer_names.size() == 0) return 1;

	// Run Network
	this->net.setInput(input_blob);
	std::vector<cv::Mat> outputs;  // outputs, layer_names
	this->net.forward(outputs, layer_names);
	input_blob.release();

	if (outputs.empty()) return 1;

	// Post-process
	std::vector<Objects> objects;

	try {
		for (int idx = 0; idx < outputs.size(); idx++) {
			Params params = this->output_layers_params[idx];
			cv::Mat output = outputs[idx];

			YoloParams layer_params(params, output.size[2]);// = new YoloParams(params, output.size[2]);

			int bbox_size = layer_params.param.coords + layer_params.param.classes + 1;

			for (int row = 0; row < layer_params.side; row++) {
				for (int col = 0; col < layer_params.side; col++) {
					for (int n = 0; n < layer_params.param.num; n++) {
						int p[4];
						p[0] = 0;
						p[2] = row;
						p[3] = col;
						//float* bbox = new float[((n + 1) * bbox_size) - (n * bbox_size)];
						/*std::vector<float> bbox;
						for (int i = 0; i < ((n + 1) * bbox_size) - (n * bbox_size); i++) {
							p[1] = i + (n * bbox_size);
							bbox.push_back(output.at<float>(&p[0]));
						}*/
						
						float x, y, width, height, object_probability;

						p[1] = n * bbox_size;
						x = output.at<float>(&p[0]);
						p[1] = n * bbox_size + 1;
						y = output.at<float>(&p[0]);
						p[1] = n * bbox_size + 2;
						width = output.at<float>(&p[0]);
						p[1] = n * bbox_size + 3;
						height = output.at<float>(&p[0]);
						p[1] = n * bbox_size + 4;
						object_probability = output.at<float>(&p[0]);

						if (object_probability < this->CONF_THRESH) {
							continue;
						}

						try {
							width = exp(width);
							height = exp(height);
						}
						catch (...) {
							continue;
						}

						//float* class_probabilities = new float[((n + 1) * bbox_size) - (n * bbox_size) - 5];
						std::vector<float> class_probabilities;
						for (int i = 0; i < ((n + 1) * bbox_size) - (n * bbox_size) - 5; i++) {
							p[1] = i + (n * bbox_size) + 5;
							class_probabilities.push_back(output.at<float>(&p[0]));
						}

						x = (col + x) / (1.0 * layer_params.side);
						y = (row + y) / (1.0 * layer_params.side);

						width = width * layer_params.anchors[2 * n] / static_cast<float>(SPATIAL_SIZE);
						height = height * layer_params.anchors[2 * n + 1] / static_cast<float>(SPATIAL_SIZE);

						int class_id = 0;
						float tmp_max = class_probabilities[0];

						for (int i = 0; i < ((n + 1) * bbox_size) - (n * bbox_size) - 5; i++) {
							if (tmp_max < class_probabilities[i]) {
								tmp_max = class_probabilities[i];
								class_id = i;
							}
						}

						float confidence = class_probabilities[class_id] * object_probability;
						if (confidence < this->CONF_THRESH) {
							continue;
						}

						int xmin = static_cast<int>(floor((x - width / 2.0) * img.size[1]));
						int ymin = static_cast<int>(floor((y - height / 2.0) * img.size[0]));
						int xmax = static_cast<int>(floor(xmin + width * img.size[1]));
						int ymax = static_cast<int>(floor(ymin + height * img.size[0]));

						Objects tmp_o;
						tmp_o.xmin = xmin;
						tmp_o.xmax = xmax;
						tmp_o.ymin = ymin;
						tmp_o.ymax = ymax;
						tmp_o.class_id = class_id;
						tmp_o.confidence = confidence;
						objects.push_back(tmp_o);
					}
				}
			}
		}
	}
	catch (...) {
		return 1;
	}

	//outputs.clear();

	for (int i = 1; i < objects.size(); i++) {
		Objects key = objects[i];
		int j = i - 1;

		while (j >= 0 && objects[j].confidence < key.confidence) {
			objects[j + 1] = objects[j];
			j = j - 1;
		}
		objects[j + 1] = key;
	}

	for (int i = 0; i < objects.size(); i++) {
		if (objects[i].confidence == 0)
			continue;
		for (int j = i + 1; j < objects.size(); j++) {
			if (this->intersection_over_union(objects[i], objects[j]) > NMS_THRESH) {
				objects[j].confidence = 0;
			}
		}
	}

	std::vector<Objects> new_objs;
	for (int i = 0; i < objects.size(); i++) {
		if (objects[i].confidence >= this->CONF_THRESH)
			new_objs.push_back(objects[i]);
	}
	//objects.clear();

	//if (new_objs.size() == 0) return 1;

	for (int i = 0; i < new_objs.size(); i++) {
		cv::Rect rect(
			static_cast<int>(std::max(new_objs[i].xmin, 0)),
			static_cast<int>(std::max(new_objs[i].ymin, 0)),
			static_cast<int>(std::min(new_objs[i].xmax, img.size[1]) - std::max(new_objs[i].xmin, 0)),
			static_cast<int>(std::min(new_objs[i].ymax, img.size[0]) - std::max(new_objs[i].ymin, 0))
		);
		openVinoRes tmp_res;
		tmp_res.coords = rect;
		tmp_res.classId = new_objs[i].class_id;
		tmp_res.confidence = new_objs[i].confidence;
		res.push_back(tmp_res);
	}
	//new_objs.clear();
	//layer_names.clear();
	return 0;
}

float OpenVinoSdet::intersection_over_union(const Objects& box_1, const Objects& box_2)
{
	float width_of_overlap_area = std::fmin(box_1.xmax, box_2.xmax) - std::fmax(box_1.xmin, box_2.xmin);
	float height_of_overlap_area = std::fmin(box_1.ymax, box_2.ymax) - std::fmax(box_1.ymin, box_2.ymin);
	float area_of_overlap;

	if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
		area_of_overlap = 0.0;
	else
		area_of_overlap = width_of_overlap_area * height_of_overlap_area;

	float box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin);
	float box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin);

	float area_of_union = box_1_area + box_2_area - area_of_overlap;
	if (area_of_union == 0)
		return 0.0;

	return area_of_overlap / area_of_union;
}
