#include "mtcnn.h"



static uint8_t* get_img(cv::Mat img) {
	uchar * colorData = new uchar[img.total() * 4];
	cv::Mat MatTemp(img.size(), CV_8UC4, colorData);
	cv::cvtColor(img, MatTemp, cv::COLOR_BGR2RGBA, 4);
	//cv::cvtColor(img, MatTemp, type, 4);
	return (uint8_t *)MatTemp.data;
}

static bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
	return a.bbox.score > b.bbox.score;
}


static float IoU(float xmin, float ymin, float xmax, float ymax,
	float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom) {
	float iw = std::min(xmax, xmax_) - std::max(xmin, xmin_) + 1;
	float ih = std::min(ymax, ymax_) - std::max(ymin, ymin_) + 1;
	if (iw <= 0 || ih <= 0)
		return 0;
	float s = iw*ih;
	if (is_iom) {
		float ov = s / std::min((xmax - xmin + 1)*(ymax - ymin + 1), (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1));
		return ov;
	}
	else {
		float ov = s / ((xmax - xmin + 1)*(ymax - ymin + 1) + (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1) - s);
		return ov;
	}
}

static float IoU(FaceInfo f1, FaceInfo f2, bool is_iom) {
	float iw = std::min(f1.bbox.xmax, f2.bbox.xmax) - std::max(f1.bbox.xmin, f2.bbox.xmin) + 1;
	float ih = std::min(f1.bbox.ymax, f2.bbox.ymax) - std::max(f1.bbox.ymin, f2.bbox.ymin) + 1;
	if (iw <= 0 || ih <= 0)
		return 0;
	float s = iw*ih;
	if (is_iom) {
		float ov = s / std::min((f1.bbox.xmax - f1.bbox.xmin + 1)*(f1.bbox.ymax - f1.bbox.ymin + 1), (f2.bbox.xmax - f2.bbox.xmin + 1)*(f2.bbox.ymax - f2.bbox.ymin + 1));
		return ov;
	}
	else {
		float ov = s / ((f1.bbox.xmax - f1.bbox.xmin + 1)*(f1.bbox.ymax - f1.bbox.ymin + 1) + (f2.bbox.xmax - f2.bbox.xmin + 1)*(f2.bbox.ymax - f2.bbox.ymin + 1) - s);
		return ov;
	}
}

static std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes,
	float thresh, char methodType) {
	std::vector<FaceInfo> bboxes_nms;
	if (bboxes.size() == 0) {
		return bboxes_nms;
	}
	std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

	int32_t select_idx = 0;
	int32_t num_bbox = static_cast<int32_t>(bboxes.size());
	std::vector<int32_t> mask_merged(num_bbox, 0);
	bool all_merged = false;

	while (!all_merged) {
		while (select_idx < num_bbox && mask_merged[select_idx] == 1)
			select_idx++;
		if (select_idx == num_bbox) {
			all_merged = true;
			continue;
		}
		bboxes_nms.push_back(bboxes[select_idx]);
		mask_merged[select_idx] = 1;

		FaceBox select_bbox = bboxes[select_idx].bbox;
		float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
		float x1 = static_cast<float>(select_bbox.xmin);
		float y1 = static_cast<float>(select_bbox.ymin);
		float x2 = static_cast<float>(select_bbox.xmax);
		float y2 = static_cast<float>(select_bbox.ymax);

		select_idx++;
		//#ifdef _OPENMP
		//#pragma omp parallel for num_threads(threads_num)
		//#endif
		for (int32_t i = select_idx; i < num_bbox; i++) {
			if (mask_merged[i] == 1)
				continue;

			FaceBox & bbox_i = bboxes[i].bbox;
			float x = std::max<float>(x1, static_cast<float>(bbox_i.xmin));
			float y = std::max<float>(y1, static_cast<float>(bbox_i.ymin));
			float w = std::min<float>(x2, static_cast<float>(bbox_i.xmax)) - x + 1;
			float h = std::min<float>(y2, static_cast<float>(bbox_i.ymax)) - y + 1;
			if (w <= 0 || h <= 0)
				continue;

			float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
			float area_intersect = w * h;

			switch (methodType) {
			case 'u':
				if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
					mask_merged[i] = 1;
				break;
			case 'm':
				if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
					mask_merged[i] = 1;
				break;
			default:
				break;
			}
		}
	}
	return bboxes_nms;
}
static void BBoxRegression(vector<FaceInfo>& bboxes) {
	//#ifdef _OPENMP
	//#pragma omp parallel for num_threads(threads_num)
	//#endif
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		float *bbox_reg = bboxes[i].bbox_reg;
		float w = bbox.xmax - bbox.xmin + 1;
		float h = bbox.ymax - bbox.ymin + 1;
		bbox.xmin += bbox_reg[0] * w;
		bbox.ymin += bbox_reg[1] * h;
		bbox.xmax += bbox_reg[2] * w;
		bbox.ymax += bbox_reg[3] * h;
	}
}
static void BBoxPad(vector<FaceInfo>& bboxes, int width, int height) {
	//#ifdef _OPENMP
	//#pragma omp parallel for num_threads(threads_num)
	//#endif
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		bbox.xmin = round(std::max(bbox.xmin, 0.f));
		bbox.ymin = round(std::max(bbox.ymin, 0.f));
		bbox.xmax = round(std::min(bbox.xmax, width - 1.f));
		bbox.ymax = round(std::min(bbox.ymax, height - 1.f));
	}
}
static void BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height) {
	//#ifdef _OPENMP
	//#pragma omp parallel for num_threads(threads_num)
	//#endif
	for (int i = 0; i < bboxes.size(); ++i) {
		FaceBox &bbox = bboxes[i].bbox;
		float w = bbox.xmax - bbox.xmin + 1;
		float h = bbox.ymax - bbox.ymin + 1;
		float side = h>w ? h : w;
		bbox.xmin = round(std::max(bbox.xmin + (w - side)*0.5f, 0.f));
		bbox.ymin = round(std::max(bbox.ymin + (h - side)*0.5f, 0.f));
		bbox.xmax = round(std::min(bbox.xmin + side - 1, width - 1.f));
		bbox.ymax = round(std::min(bbox.ymin + side - 1, height - 1.f));
	}
}


void MTCNN::GenerateBBox(float * confidence_data, float *reg_box, int feature_map_w_, int feature_map_h_, float scale, float thresh) {

	int spatical_size = feature_map_w_*feature_map_h_;

	candidate_boxes_.clear();
	float v_scale = 1.0 / scale;
	for (int i = 0; i<spatical_size; ++i) {
		int stride = i << 2;
		if (confidence_data[stride + 1] >= thresh) {
			int y = i / feature_map_w_;
			int x = i - feature_map_w_ * y;
			FaceInfo faceInfo;
			FaceBox &faceBox = faceInfo.bbox;

			faceBox.xmin = (float)(x * pnet_stride) * v_scale;
			faceBox.ymin = (float)(y * pnet_stride) * v_scale;
			faceBox.xmax = (float)(x * pnet_stride + pnet_cell_size - 1.f) * v_scale;
			faceBox.ymax = (float)(y * pnet_stride + pnet_cell_size - 1.f) * v_scale;

			faceInfo.bbox_reg[0] = reg_box[stride];
			faceInfo.bbox_reg[1] = reg_box[stride + 1];
			faceInfo.bbox_reg[2] = reg_box[stride + 2];
			faceInfo.bbox_reg[3] = reg_box[stride + 3];

			faceBox.score = confidence_data[stride + 1];
			candidate_boxes_.push_back(faceInfo);
		}
	}
}

MTCNN::MTCNN() {
}

void MTCNN::init(const string& proto_model_dir){

	PNet_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile((proto_model_dir + "/det1.mnn").c_str()));

	RNet_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile((proto_model_dir + "/det2.mnn").c_str()));

	ONet_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile((proto_model_dir + "/det3-half.mnn").c_str()));


	MNN::ScheduleConfig config;
	config.type = (MNNForwardType)0;
	config.numThread = 1; // 1 faster

	BackendConfig backendConfig;
	backendConfig.precision = BackendConfig::Precision_Low;
	backendConfig.power = BackendConfig::Power_High;
	config.backendConfig = &backendConfig;

	sess_p = PNet_->createSession(config);
	sess_r = RNet_->createSession(config);
	sess_o = ONet_->createSession(config);


	p_input = PNet_->getSessionInput(sess_p, NULL);
	p_out_pro = PNet_->getSessionOutput(sess_p, "prob1");
	p_out_reg = PNet_->getSessionOutput(sess_p, "conv4-2");

	r_input = RNet_->getSessionInput(sess_r, NULL);
	r_out_pro = RNet_->getSessionOutput(sess_r, "prob1");
	r_out_reg = RNet_->getSessionOutput(sess_r, "conv5-2");

	o_input = ONet_->getSessionInput(sess_o, NULL);
	o_out_pro = ONet_->getSessionOutput(sess_o, "prob1");
	o_out_reg = ONet_->getSessionOutput(sess_o, "conv6-2");
	o_out_lank = ONet_->getSessionOutput(sess_o, "conv6-3");

	ImageProcess::Config config_data;
	config_data.filterType = BILINEAR;
	const float mean_vals[3] = { mean_val, mean_val, mean_val };
	const float norm_vals[3] = { std_val, std_val, std_val };
	::memcpy(config_data.mean, mean_vals, sizeof(mean_vals));
	::memcpy(config_data.normal, norm_vals, sizeof(norm_vals));
	config_data.sourceFormat = RGBA;
	config_data.destFormat = RGB;

	pretreat_data = std::shared_ptr<ImageProcess>(ImageProcess::create(config_data));
}


MTCNN::~MTCNN() {
	PNet_->releaseModel();
	RNet_->releaseModel();
	ONet_->releaseModel();
	candidate_boxes_.clear();
	total_boxes_.clear();
}

void MTCNN::SetSmooth(bool smooth1)
{
	smooth = smooth1;
}

bool MTCNN::GetSmooth()
{
	return smooth;
}


void MTCNN::setThrehold(float threhold[3])
{
	threhold_p = threhold[0];
	threhold_r = threhold[1];
	threhold_o = threhold[2];
}

void MTCNN::setFactor(float factor1)
{
	factor = factor1;
}

void MTCNN::setThreadNum(int threads_num1)
{
	threads_num = threads_num1;
}

void MTCNN::setIsMaxFace(bool isMaxFace1)
{
	isMaxFace = isMaxFace1;
}

void MTCNN::setMinFace(int min_face1)
{
	min_face = min_face1;
}

vector<FaceInfo> MTCNN::ProposalNet(const cv::Mat& img, int minSize, float threshold, float factor) {

	int width = img.cols;
	int height = img.rows;
	float scale = 12.0f / minSize;
	float minWH = std::min(height, width) *scale;
	std::vector<float> scales;
	while (minWH >= 12) {
		scales.push_back(scale);
		minWH *= factor;
		scale *= factor;
	}
	total_boxes_.clear();

	uint8_t *pImg = get_img(img);
	for (int i = 0; i < scales.size(); i++) {
		int ws = (int)std::ceil(width*scales[i]);
		int hs = (int)std::ceil(height*scales[i]);


		std::vector<int> inputDims = { 1, 3, hs, ws };
		PNet_->resizeTensor(p_input, inputDims);
		PNet_->resizeSession(sess_p);

		MNN::CV::Matrix trans;
		trans.postScale(1.0f / ws, 1.0f / hs);
		trans.postScale(width, height);
		pretreat_data->setMatrix(trans);
		pretreat_data->convert(pImg, width, height, 0, p_input);


		PNet_->runSession(sess_p);
		float * confidence = p_out_pro->host<float>();
		float * reg = p_out_reg->host<float>();

		int feature_w = p_out_pro->width();
		int feature_h = p_out_pro->height();

		GenerateBBox(confidence, reg, feature_w, feature_h, scales[i], threshold);
		std::vector<FaceInfo> bboxes_nms = NMS(candidate_boxes_, 0.5f, 'u');
		if (bboxes_nms.size() > 0) {
			total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
		}
	}

	int num_box = (int)total_boxes_.size();
	vector<FaceInfo> res_boxes;
	if (num_box != 0) {
		res_boxes = NMS(total_boxes_, 0.5f, 'u');
		BBoxRegression(res_boxes);
		BBoxPadSquare(res_boxes, width, height);
	}
	delete pImg;
	return res_boxes;
}

std::vector<FaceInfo> MTCNN::NextStage(const cv::Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold) {
	vector<FaceInfo> res;
	int batch_size = pre_stage_res.size();

	switch (stage_num) {
	case 2: {

		for (int n = 0; n < batch_size; ++n)
		{
			FaceBox &box = pre_stage_res[n].bbox;
			cv::Mat roi = image(cv::Rect(cv::Point((int)box.xmin, (int)box.ymin), cv::Point((int)box.xmax, (int)box.ymax))).clone();

			//cv::imshow("face", roi);
			//cv::waitKey(0);

			MNN::CV::Matrix trans;
			trans.postScale(1.0 / input_w, 1.0 / input_h);
			trans.postScale(roi.cols, roi.rows);
			pretreat_data->setMatrix(trans);

			uint8_t *pImg = get_img(roi);
			pretreat_data->convert(pImg, roi.cols, roi.rows, 0, r_input);
			delete pImg;
			RNet_->runSession(sess_r);

			float * confidence = r_out_pro->host<float>();
			float * reg_box = r_out_reg->host<float>();

			float conf = confidence[1];
			if (conf >= threshold) {
				FaceInfo info;
				info.bbox.score = conf;
				info.bbox.xmin = pre_stage_res[n].bbox.xmin;
				info.bbox.ymin = pre_stage_res[n].bbox.ymin;
				info.bbox.xmax = pre_stage_res[n].bbox.xmax;
				info.bbox.ymax = pre_stage_res[n].bbox.ymax;
				for (int i = 0; i < 4; ++i) {
					info.bbox_reg[i] = reg_box[i];
				}
				res.push_back(info);

			}
		}
		break;
	}
	case 3: {
		//#ifdef _OPENMP
		//#pragma omp parallel for num_threads(threads_num)
		//#endif
		for (int n = 0; n < batch_size; ++n)
		{
			FaceBox &box = pre_stage_res[n].bbox;
			cv::Mat roi = image(cv::Rect(cv::Point((int)box.xmin, (int)box.ymin), cv::Point((int)box.xmax, (int)box.ymax))).clone();

			//cv::imshow("face", roi);
			//cv::waitKey(0);

			MNN::CV::Matrix trans;
			trans.postScale(1.0f / input_w, 1.0f / input_h);
			trans.postScale(roi.cols, roi.rows);
			pretreat_data->setMatrix(trans);
			uint8_t *pImg = get_img(roi);
			pretreat_data->convert(pImg, roi.cols, roi.rows, 0, o_input);
			delete pImg;
			ONet_->runSession(sess_o);
			float * confidence = o_out_pro->host<float>();
			float * reg_box = o_out_reg->host<float>();
			float * reg_landmark = o_out_lank->host<float>();

			float conf = confidence[1];
			//std::cout<<"stage three:"<<confidence[0]<<" "<<confidence[1]<<" "<<confidence[2]<<" "<<confidence[4]<<std::endl;
			if (conf >= threshold) {
				FaceInfo info;
				info.bbox.score = conf;
				info.bbox.xmin = pre_stage_res[n].bbox.xmin;
				info.bbox.ymin = pre_stage_res[n].bbox.ymin;
				info.bbox.xmax = pre_stage_res[n].bbox.xmax;
				info.bbox.ymax = pre_stage_res[n].bbox.ymax;
				for (int i = 0; i < 4; ++i) {
					info.bbox_reg[i] = reg_box[i];
				}
				float w = info.bbox.xmax - info.bbox.xmin + 1.f;
				float h = info.bbox.ymax - info.bbox.ymin + 1.f;
				for (int i = 0; i < 5; ++i) {
					info.landmark[i] = reg_landmark[2 * i] * w + info.bbox.xmin;
					info.landmark[5 + i] = reg_landmark[2 * i + 1] * h + info.bbox.ymin;
				}
				res.push_back(info);
			}
		}
		break;
	}
	default:
		return res;
		break;
	}
	return res;
}


static void SmoothBbox(std::vector<FaceInfo>& finalBbox)
{
	static std::vector<FaceInfo> preBbox_;
	for (int i = 0; i < finalBbox.size(); i++) {
		for (int j = 0; j < preBbox_.size(); j++) {
			//if (IoU(finalBbox[i], preBbox_[j], false) > 0.9)
			//	finalBbox[i] = preBbox_[j];
			//else if (IoU(finalBbox[i], preBbox_[j], false) > 0.6) {
			if (IoU(finalBbox[i], preBbox_[j], false) > 0.85) {
				finalBbox[i].bbox.xmin = (finalBbox[i].bbox.xmin + preBbox_[j].bbox.xmin) / 2;
				finalBbox[i].bbox.ymin = (finalBbox[i].bbox.ymin + preBbox_[j].bbox.ymin) / 2;
				finalBbox[i].bbox.xmax = (finalBbox[i].bbox.xmax + preBbox_[j].bbox.xmax) / 2;
				finalBbox[i].bbox.ymax = (finalBbox[i].bbox.ymax + preBbox_[j].bbox.ymax) / 2;
				//finalBbox[i].area = (finalBbox[i].x2 - finalBbox[i].x1)*(finalBbox[i].y2 - finalBbox[i].y1);
				for (int k = 0; k < 10; k++)
				{
					finalBbox[i].landmark[k] = (finalBbox[i].landmark[k] + preBbox_[j].landmark[k]) / 2;
				}
			}
		}
	}
	preBbox_ = finalBbox;

}

void MTCNN::Detect_T(const cv::Mat & img, std::vector<FaceInfo>& result)
{
	if (isMaxFace)
	{
		result = Detect_MaxFace(img);
	}
	else
	{
		result = Detect(img);
	}
}

vector<FaceInfo> MTCNN::Detect(const cv::Mat& image, const int stage) {

	vector<FaceInfo> pnet_res;
	vector<FaceInfo> rnet_res;
	vector<FaceInfo> onet_res;

	if (stage >= 1) {
		pnet_res = ProposalNet(image, min_face, threhold_p, factor);
	}
	//std::cout<<"p size is:"<<pnet_res.size()<<std::endl;
	if (stage >= 2 && pnet_res.size()>0) {
		if (pnet_max_detect_num < (int)pnet_res.size()) {
			pnet_res.resize(pnet_max_detect_num);
		}
		rnet_res = NextStage(image, pnet_res, 24, 24, 2, threhold_r);
		rnet_res = NMS(rnet_res, iou_threhold, 'u');
		BBoxRegression(rnet_res);
		BBoxPadSquare(rnet_res, image.cols, image.rows);
	}
	//std::cout<<"r size is:"<<rnet_res.size()<<std::endl;
	if (stage >= 3 && rnet_res.size()>0) {
		onet_res = NextStage(image, rnet_res, 48, 48, 3, threhold_o);
		BBoxRegression(onet_res);
		onet_res = NMS(onet_res, iou_threhold, 'm');
		BBoxPad(onet_res, image.cols, image.rows);
	}
	if (stage == 1) {
		if (smooth)
			SmoothBbox(pnet_res);
		return pnet_res;
	}
	else if (stage == 2) {
		if (smooth)
			SmoothBbox(pnet_res);
		return rnet_res;
	}
	else if (stage == 3) {
		if (smooth)
			SmoothBbox(onet_res);
		return onet_res;
	}
	else {
		if (smooth)
			SmoothBbox(onet_res);
		return onet_res;
	}
}


static std::vector<FaceInfo> extractMaxFace(std::vector<FaceInfo> boundingBox_)
{
	if (boundingBox_.empty()) {
		return std::vector<FaceInfo>{};
	}
	/*
	sort(boundingBox_.begin(), boundingBox_.end(), CompareBBox);
	for (std::vector<FaceInfo>::iterator itx = boundingBox_.begin() + 1; itx != boundingBox_.end();) {
	itx = boundingBox_.erase(itx);
	}
	*/
	float max_area = 0;
	int index = 0;
	for (int i = 0; i < boundingBox_.size(); ++i) {
		FaceBox select_bbox = boundingBox_[i].bbox;
		float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
		if (area1 > max_area) {
			max_area = area1;
			index = i;
		}
	}
	return std::vector<FaceInfo>{boundingBox_[index]};
}

std::vector<FaceInfo> MTCNN::Detect_MaxFace(const cv::Mat& img, const int stage) {
	vector<FaceInfo> pnet_res;
	vector<FaceInfo> rnet_res;
	vector<FaceInfo> onet_res;

	//total_boxes_.clear();
	//candidate_boxes_.clear();

	int width = img.cols;
	int height = img.rows;
	float scale = 12.0f / min_face;
	float minWH = std::min(height, width) *scale;
	std::vector<float> scales;
	while (minWH >= 12) {
		scales.push_back(scale);
		minWH *= factor;
		scale *= factor;
	}

	//sort(scales.begin(), scales.end());
	std::reverse(scales.begin(), scales.end());
	
	uint8_t *pImg = get_img(img);
	for (int i = 0; i < scales.size(); i++) {
		int ws = (int)std::ceil(width*scales[i]);
		int hs = (int)std::ceil(height*scales[i]);
		std::vector<int> inputDims = { 1, 3, hs, ws };
		PNet_->resizeTensor(p_input, inputDims);
		PNet_->resizeSession(sess_p);

		MNN::CV::Matrix trans;
		trans.postScale(1.0f / ws, 1.0f / hs);
		trans.postScale(width, height);
		pretreat_data->setMatrix(trans);
		pretreat_data->convert(pImg, width, height, 0, p_input);

		PNet_->runSession(sess_p);
		float * confidence = p_out_pro->host<float>();
		float * reg = p_out_reg->host<float>();

		int feature_w = p_out_pro->width();
		int feature_h = p_out_pro->height();

		GenerateBBox(confidence, reg, feature_w, feature_h, scales[i], threhold_p);
		std::vector<FaceInfo> bboxes_nms = NMS(candidate_boxes_, 0.5f, 'u');

		//nmsTwoBoxs(bboxes_nms, pnet_res, 0.5);
		if (bboxes_nms.size() > 0) {
			pnet_res.insert(pnet_res.end(), bboxes_nms.begin(), bboxes_nms.end());
		}
		else {
			continue;
		}
		BBoxRegression(pnet_res);
		BBoxPadSquare(pnet_res, width, height);

		bboxes_nms.clear();
		bboxes_nms = NextStage(img, pnet_res, 24, 24, 2, threhold_r);
		bboxes_nms = NMS(bboxes_nms, iou_threhold, 'u');
		//nmsTwoBoxs(bboxes_nms, rnet_res, 0.5)
		if (bboxes_nms.size() > 0) {
			rnet_res.insert(rnet_res.end(), bboxes_nms.begin(), bboxes_nms.end());
		}
		else {
			pnet_res.clear();
			continue;
		}
		BBoxRegression(rnet_res);
		BBoxPadSquare(rnet_res, img.cols, img.rows);


		onet_res = NextStage(img, rnet_res, 48, 48, 3, threhold_r);

		BBoxRegression(onet_res);
		onet_res = NMS(onet_res, iou_threhold, 'm');
		BBoxPad(onet_res, img.cols, img.rows);

		if (onet_res.size() < 1) {
			pnet_res.clear();
			rnet_res.clear();
			continue;
		}
		else {
			onet_res = extractMaxFace(onet_res);
			delete pImg;
			if (smooth)
				SmoothBbox(onet_res);
			return onet_res;
		}
	}
	delete pImg;
	return std::vector<FaceInfo>{};
}

float MTCNN::rnet(cv::Mat & image, cv::Rect & face_)
{
	face_ = cv::Rect(0, 0, image.cols, image.rows)&face_;
	cv::Mat roi = image(face_).clone();

	//cv::imshow("face", roi);
	//cv::waitKey(0);

	MNN::CV::Matrix trans;
	trans.postScale(1.0 / 24, 1.0 / 24);
	trans.postScale(roi.cols, roi.rows);
	pretreat_data->setMatrix(trans);

	uint8_t *pImg = get_img(roi);
	pretreat_data->convert(pImg, roi.cols, roi.rows, 0, r_input);
	delete pImg;
	RNet_->runSession(sess_r);

	float * confidence = r_out_pro->host<float>();
	//float * reg_box = r_out_reg->host<float>();

	float conf = confidence[1];
	return conf;
}

FaceInfo MTCNN::onet(cv::Mat & image, cv::Rect & face_)
{
	FaceInfo faceInfos;

	face_ = cv::Rect(0, 0, image.cols, image.rows)&face_;
	cv::Mat roi = image(face_).clone();

	//cv::imshow("face", roi);
	//cv::waitKey(0);

	MNN::CV::Matrix trans;
	trans.postScale(1.0 / 48, 1.0 / 48);
	trans.postScale(roi.cols, roi.rows);
	pretreat_data->setMatrix(trans);

	uint8_t *pImg = get_img(roi);
	pretreat_data->convert(pImg, roi.cols, roi.rows, 0, o_input);
	delete pImg;
	ONet_->runSession(sess_o);
	float * confidence = o_out_pro->host<float>();
	float * reg_box = o_out_reg->host<float>();
	float * reg_landmark = o_out_lank->host<float>();

	float conf = confidence[1];
	//std::cout<<"stage three:"<<confidence[0]<<" "<<confidence[1]<<" "<<confidence[2]<<" "<<confidence[4]<<std::endl;
	//if (conf >= threhold_o) {

		faceInfos.bbox.score = conf;
		faceInfos.bbox.xmin = face_.x;
		faceInfos.bbox.ymin = face_.y;
		faceInfos.bbox.xmax = face_.x + face_.width;
		faceInfos.bbox.ymax = face_.y + face_.height;
		for (int i = 0; i < 4; ++i) {
			faceInfos.bbox_reg[i] = reg_box[i];
		}
		float w = faceInfos.bbox.xmax - faceInfos.bbox.xmin + 1.f;
		float h = faceInfos.bbox.ymax - faceInfos.bbox.ymin + 1.f;
		for (int i = 0; i < 5; ++i) {
			faceInfos.landmark[i] = reg_landmark[2 * i] * w + faceInfos.bbox.xmin;
			faceInfos.landmark[5 + i] = reg_landmark[2 * i + 1] * h + faceInfos.bbox.ymin;
		}
	//}
	return faceInfos;
}
