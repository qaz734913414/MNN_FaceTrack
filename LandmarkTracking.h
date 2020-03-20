#ifndef ZEUSEESFACETRACKING_H
#define ZEUSEESFACETRACKING_H

#include <opencv2/opencv.hpp>
#include "mtcnn.h"
#include "time.h"

cv::Rect boundingRect(const std::vector<cv::Point>& pts) {
	if (pts.size() > 1)
	{
		int xmin = pts[0].x;
		int ymin = pts[0].y;
		int xmax = pts[0].x;
		int ymax = pts[0].y;
		for (int i = 1; i < pts.size(); i++)
		{
			if (pts[i].x < xmin)
				xmin = pts[i].x;
			if (pts[i].y < ymin)
				ymin = pts[i].y;
			if (pts[i].x > xmax)
				xmax = pts[i].x;
			if (pts[i].y > ymax)
				ymax = pts[i].y;
		}
		return cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
	}
}


//typedef int T;
//T i = 1;


class Face {
public:

	Face(int instance_id,cv::Rect loc_t) {
		face_id = instance_id;
		isCanShow = false; //追踪一次后待框稳定后即可显示
		loc = loc_t;
	}

	Face() {

		isCanShow = false; //追踪一次后待框稳定后即可显示
	}

	FaceInfo faceBbox;

	int face_id = -1;
	long frameId = 0;
	int ptr_num = 0;

	bool isCanShow;
	cv::Mat frame_face_prev;
	cv::Rect loc;
	//int loseNum = 0;

	static cv::Rect SquarePadding(cv::Rect facebox, int margin_rows, int margin_cols, bool max_b)
	{
		int c_x = facebox.x + facebox.width / 2;
		int c_y = facebox.y + facebox.height / 2;
		int large = 0;
		if (max_b)
			large = max(facebox.height, facebox.width) / 2;
		else
			large = min(facebox.height, facebox.width) / 2;
		cv::Rect rectNot(c_x - large, c_y - large, c_x + large, c_y + large);
		rectNot.x = max(0, rectNot.x);
		rectNot.y = max(0, rectNot.y);
		rectNot.height = min(rectNot.height, margin_rows - 1);
		rectNot.width = min(rectNot.width, margin_cols - 1);
		if (rectNot.height - rectNot.y != rectNot.width - rectNot.x)
			return SquarePadding(cv::Rect(rectNot.x, rectNot.y, rectNot.width - rectNot.x, rectNot.height - rectNot.y), margin_rows, margin_cols, false);

		return cv::Rect(rectNot.x, rectNot.y, rectNot.width - rectNot.x, rectNot.height - rectNot.y);
	}

	static cv::Rect SquarePadding(cv::Rect facebox, int padding)
	{

		int c_x = facebox.x - padding;
		int c_y = facebox.y - padding;
		return cv::Rect(facebox.x - padding, facebox.y - padding, facebox.width + padding * 2, facebox.height + padding * 2);;
	}

	static double getDistance(cv::Point x, cv::Point y)
	{
		return sqrt((x.x - y.x) * (x.x - y.x) + (x.y - y.y) * (x.y - y.y));
	}


};



class FaceTracking {
public:
	FaceTracking(std::string modelPath)
	{
		this->detector = new MTCNN();
		this->detector->init(modelPath);
		this->detector->setIsMaxFace(isMaxFace);
		faceMinSize = 80;
		this->detector->setMinFace(faceMinSize);
		detection_Time = -1;

	}

	~FaceTracking() {
		delete this->detector;

	}

	void detecting(cv::Mat& image) {

		std::vector<FaceInfo> finalBbox;
		detector->Detect_T(image, finalBbox);
		
		const int num_box = finalBbox.size();
		std::vector<cv::Rect> bbox;
		bbox.resize(num_box);
		candidateFaces_lock = 1;
		for (int i = 0; i < num_box; i++) {
			bbox[i] = cv::Rect(finalBbox[i].bbox.xmin, finalBbox[i].bbox.ymin, finalBbox[i].bbox.xmax - finalBbox[i].bbox.xmin + 1,
				finalBbox[i].bbox.ymax - finalBbox[i].bbox.ymin + 1);
			bbox[i] = Face::SquarePadding(bbox[i], image.rows, image.cols, true);
			
			std::shared_ptr<Face> face(new Face(trackingID, bbox[i]));

			image(bbox[i]).copyTo(face->frame_face_prev);

			trackingID = trackingID + 1;
			candidateFaces.push_back(*face);
		}
		candidateFaces_lock = 0;
	}

	void Init(cv::Mat& image) {
		ImageHighDP = image.clone();
		trackingID = 0;
		detection_Interval = 150; //detect faces every 200 ms
		detecting(image);
		stabilization = false;

	}

	void doingLandmark_onet(cv::Mat& img, FaceInfo& faceBbox, cv::Rect &face_roi, int stable_state = 0) {
		
		faceBbox = detector->onet(img, face_roi);

	}



	void tracking_corrfilter(const cv::Mat& frame, const cv::Mat& model, cv::Rect& trackBox, float scale)
	{
		trackBox.x /= scale;
		trackBox.y /= scale;
		trackBox.height /= scale;
		trackBox.width /= scale;
		int zeroadd_x = 0;
		int zeroadd_y = 0;
		cv::Mat frame_;
		cv::Mat model_;
		cv::resize(frame, frame_, cv::Size(), 1 / scale, 1 / scale);
		cv::resize(model, model_, cv::Size(), 1 / scale, 1 / scale);
		cv::Mat gray;
		cvtColor(frame_, gray, cv::COLOR_RGB2GRAY);
		cv::Mat gray_model;
		cvtColor(model_, gray_model, cv::COLOR_RGB2GRAY);
		cv::Rect searchWindow;
		searchWindow.width = trackBox.width * 3;
		searchWindow.height = trackBox.height * 3;
		searchWindow.x = trackBox.x + trackBox.width * 0.5 - searchWindow.width * 0.5;
		searchWindow.y = trackBox.y + trackBox.height * 0.5 - searchWindow.height * 0.5;
		searchWindow &= cv::Rect(0, 0, frame_.cols, frame_.rows);
		cv::Mat similarity;
		matchTemplate(gray(searchWindow), gray_model, similarity, cv::TM_CCOEFF_NORMED);
		double mag_r;
		cv::Point point;
		minMaxLoc(similarity, 0, &mag_r, 0, &point);
		trackBox.x = point.x + searchWindow.x;
		trackBox.y = point.y + searchWindow.y;
		trackBox.x *= scale;
		trackBox.y *= scale;
		trackBox.height *= scale;
		trackBox.width *= scale;
	}

	bool tracking(cv::Mat& image, Face& face)
	{
		cv::Rect faceROI = face.loc;
		//cv::Mat faceROI_Image;
		tracking_corrfilter(image, face.frame_face_prev, faceROI, tpm_scale);
		
		//image(faceROI).copyTo(faceROI_Image);

		FaceInfo temp;

		doingLandmark_onet(image, temp, faceROI, face.frameId > 1);

		//float sim = detector->rnet(image, faceROI);
		
		float sim = temp.bbox.score;
		
		if (sim > 0.1) {
			//stablize
			//float diff_x = 0;
			//float diff_y = 0;
			cv::Rect bdbox;

			bdbox.x = temp.bbox.xmin;
			bdbox.y = temp.bbox.ymin;
			bdbox.width = temp.bbox.xmax - temp.bbox.xmin;
			bdbox.height = temp.bbox.ymax - temp.bbox.ymin;

			//bdbox = Face::SquarePadding(bdbox, static_cast<int>(bdbox.height * -0.05));
			bdbox = Face::SquarePadding(bdbox, image.rows, image.cols, 1);

			face.faceBbox = temp;

			face.loc = bdbox;
			//face.loseNum = 0;
			image(face.loc).copyTo(face.frame_face_prev);
			face.frameId += 1;
			face.isCanShow = true;

			return true;
		}
		else
		{
			//face.loseNum++;
			return false;
		}

		

	}
	void setMask(cv::Mat& image, cv::Rect& rect_mask)
	{

		int height = image.rows;
		int width = image.cols;
		cv::Mat subImage = image(rect_mask);
		subImage.setTo(0);
	}

	void update(cv::Mat& image)
	{
		ImageHighDP = image.clone();
		//std::cout << trackingFace.size() << std::endl;
		if (candidateFaces.size() > 0 && !candidateFaces_lock)
		{
			for (int i = 0; i < candidateFaces.size(); i++)
			{
				trackingFace.push_back(candidateFaces[i]);
			}
			candidateFaces.clear();
		}
		for (std::vector<Face>::iterator iter = trackingFace.begin(); iter != trackingFace.end();)
		{
			if (!tracking(image, *iter) /*&& iter->loseNum == 5*/)
			{
				iter = trackingFace.erase(iter); //追踪失败 则删除此人脸
			}
			else {
				iter++;
			}
		}

		if (detection_Time < 0)
		{
			detection_Time = (double)cv::getTickCount();
		}
		else {
			double diff = (double)(cv::getTickCount() - detection_Time) * 1000 / cv::getTickFrequency();
			if (diff > detection_Interval)
			{
				//set Mask to protect the tracking face not to be detected.
				for (auto& face : trackingFace)
				{
					setMask(ImageHighDP, face.loc);
				}
				detection_Time = (double)cv::getTickCount();
				// do detection in thread
				detecting(ImageHighDP);
			}

		}
	}



	std::vector<Face> trackingFace; //跟踪中的人脸

private:

	//int isLostDetection;
	//int isTracking;
	//int isDetection;
	cv::Mat ImageHighDP;

	int faceMinSize;
	MTCNN* detector;
	std::vector<Face> candidateFaces; // 将检测到的人脸放入此列队 待跟踪的人脸
	bool candidateFaces_lock;
	double detection_Time;
	double detection_Interval;
	int trackingID;
	bool stabilization;
	int tpm_scale = 2;
	bool isMaxFace = true;
};
#endif //ZEUSEESFACETRACKING_H
