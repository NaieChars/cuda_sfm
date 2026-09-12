#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

#include "PointCloudExport.h"

struct FeatureSet
{
	static constexpr int DESC_DIM = 128;

	int numFeatures = 0;

	std::vector<float> kpX;	
	std::vector<float> kpY;
	std::vector<float> descriptors;	

	std::vector<cv::KeyPoint> cvKeypoints;

	//cv::Mat image;	// 保存原图
};

struct PnPCorrespondence
{
	int landmarkIdx;	// 3D 点在pointCloud中的下标，从track的value位提取
	int kpCur;			// 3D点在当前图像中对应的2D特点下标
};

// 同一对匹配点在两张图中的特征点下标
struct NewMatch
{
	int kpPrev;
	int kpCur;
};


struct CameraPose 
{
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);
};

// 一次观测：相机cameraIdx看到了点pointIdx，在该相机图像里的去畸变归一化坐标是uv
struct Observation 
{
    int cameraIdx;  // 哪个相机的观测
    int pointIdx;   // 看到的3D点在 pointCloud 中的下标
    cv::Point2f uv; // 对应在该图下的坐标
};


struct SFMResult 
{
    std::vector<CameraPose> cameras;        // 每个视角的位姿(相机0固定为单位位姿)
    std::vector<ColoredPoint3D> pointCloud; // 重建出的点云
    std::vector<Observation> observations;  // 所有观测记录，供BA使用
};

struct PnPResult
{
    cv::Mat R;
    cv::Mat t;

    std::vector<int> inlierIndiecs; // 对应输入参数 objectPoints / imagePoints 的经PnP计算后被认为是内点的下标
    bool success = false;
};