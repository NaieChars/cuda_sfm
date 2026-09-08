#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "CameraIntrinsics.h"

std::vector<cv::Point3f> cudaTriangulation(
    const std::vector<cv::Point2f>& inlierPoints1,
    const std::vector<cv::Point2f>& inlierPoints2,
    const CameraIntrinsics& intr,
    const cv::Mat& R, const cv::Mat& t, std::vector<int>& Mask3D);