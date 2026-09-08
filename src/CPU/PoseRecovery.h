#pragma once
#include <opencv2/opencv.hpp>
#include "CameraIntrinsics.h"

bool recoverCameraPose(const cv::Mat3d& E, 
    const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
    const CameraIntrinsics& intr,
    const std::vector<uchar>& essentialMask,
    cv::Mat& R, cv::Mat& t, int& inlierCount, std::vector<uchar>& poseMask);
