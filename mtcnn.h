#ifndef _FACE_DETECT_H_
#define _FACE_DETECT_H_

#include <opencv2/opencv.hpp>

#include <cmath>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>  // std::min, std::max, std::sort
#include <map>
#include <memory>

using namespace std;

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>

using namespace MNN;
using namespace MNN::CV;

typedef struct FaceBox {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
} FaceBox;
typedef struct FaceInfo {
	float bbox_reg[4];
	float landmark_reg[10];
	float landmark[10];
	FaceBox bbox;
} FaceInfo;


class MTCNN {
public:
	//MTCNN(const string& proto_model_dir);
	//MTCNN();
	MTCNN();
	void init(const string& proto_model_dir);
	void Detect_T(const cv::Mat& img, std::vector<FaceInfo>& result);
	

	float rnet(cv::Mat& image, cv::Rect& face_);

	FaceInfo onet(cv::Mat& image, cv::Rect& face_);

	~MTCNN();

	void SetSmooth(bool smooth);
	bool GetSmooth();
	void setThrehold(float threhold[3]);
	void setFactor(float factor);
	void setThreadNum(int threads_num);
	void setIsMaxFace(bool isMaxFace);
	void setMinFace(int min_face);

	

private:
	vector<FaceInfo> ProposalNet(const cv::Mat& img, int minSize, float threshold, float factor);
	void GenerateBBox(float * confidence_data, float *reg_box, int feature_map_w_, int feature_map_h_, float scale, float thresh);
	std::vector<FaceInfo> NextStage(const cv::Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
	std::vector<FaceInfo> Detect(const cv::Mat& img, const int stage = 3);
	std::vector<FaceInfo> Detect_MaxFace(const cv::Mat& img, const int stage = 3);

	std::shared_ptr<MNN::Interpreter> PNet_ = NULL;
	std::shared_ptr<MNN::Interpreter> RNet_ = NULL;
	std::shared_ptr<MNN::Interpreter> ONet_ = NULL;

	MNN::Session * sess_p = NULL;
	MNN::Session * sess_r = NULL;
	MNN::Session * sess_o = NULL;

	MNN::Tensor * p_input = nullptr;
	MNN::Tensor * p_out_pro = nullptr;
	MNN::Tensor * p_out_reg = nullptr;

	MNN::Tensor * r_input = nullptr;
	MNN::Tensor * r_out_pro = nullptr;
	MNN::Tensor * r_out_reg = nullptr;

	MNN::Tensor * o_input = nullptr;
	MNN::Tensor * o_out_pro = nullptr;
	MNN::Tensor * o_out_reg = nullptr;
	MNN::Tensor * o_out_lank = nullptr;

	std::shared_ptr<ImageProcess> pretreat_data;

	std::vector<FaceInfo> candidate_boxes_;
	std::vector<FaceInfo> total_boxes_;

	float threhold_p = 0.6f;
	float threhold_r = 0.7f;
	float threhold_o = 0.7f;
	float iou_threhold = 0.7f;
	float factor = 0.709f;
	int min_face = 80;

	//pnet config
	const float pnet_stride = 2;
	const float pnet_cell_size = 12;
	const int pnet_max_detect_num = 5000;
	//mean & std
	const float mean_val = 127.5f;
	const float std_val = 0.0078125f;


    int threads_num = 2;
	bool smooth = true;//Œ»∂®–‘
	bool isMaxFace = true;

};










#endif // _FaceDetect_H_

