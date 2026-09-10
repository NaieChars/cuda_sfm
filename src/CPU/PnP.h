#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include "CameraIntrinsics.h"

struct PnPResult
{
    cv::Mat R;
    cv::Mat t;

    std::vector<int> inlierIndiecs; // 对应输入参数 objectPoints / imagePoints 的经PnP计算后被认为是内点的下标
    bool success = false;
};


// 使用已有3D点 + 他们对于当前图像中对应的2D点，求新相机在世界坐标系下的位姿
PnPResult solvePnPForNewView(
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints,
    const CameraIntrinsics& intr
);