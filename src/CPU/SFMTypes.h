#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "PointCloudExport.h"

struct CameraPose 
{
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat t = cv::Mat::zeros(3, 1, CV_32F);
};

struct SFMResult 
{
    std::vector<CameraPose> cameras;        // 每个视角的位姿(相机0固定为单位位姿)
    std::vector<ColoredPoint3D> pointCloud; // 重建出的点云
    //std::vector<Observation> observations;  // 所有观测记录，供BundleAdjustment使用
};