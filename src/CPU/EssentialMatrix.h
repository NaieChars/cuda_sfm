#pragma once
#include <opencv2/opencv.hpp>
#include "CameraIntrinsics.h"

cv::Mat estimateEssentialMatrix(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
        const CameraIntrinsics& intr,std::vector<uchar>& essentialMask);