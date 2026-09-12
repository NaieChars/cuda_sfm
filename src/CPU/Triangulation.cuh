#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "CameraIntrinsics.h"
#include "DataStruct.h"

std::vector<cv::Point3f> cudaTriangulation(
    const std::vector<cv::Point2f>& inlierPoints1,
    const std::vector<cv::Point2f>& inlierPoints2,
    const CameraIntrinsics& intr,
    const CameraPose& pose1,
    const CameraPose& pose2, 
    std::vector<int>& Mask3D);