#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include "CameraIntrinsics.h"
#include "DataStruct.h"


// 使用已有3D点 + 他们对于当前图像中对应的2D点，求新相机在世界坐标系下的位姿
PnPResult solvePnPForNewView(
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints,
    const CameraIntrinsics& intr
);