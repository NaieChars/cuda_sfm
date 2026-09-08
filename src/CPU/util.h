#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// 根据求掩码 Mask 从特征点中进行筛选
std::vector<cv::Point2f>fliterInlierPoints(const std::vector<cv::Point2f>& points, const std::vector<uchar>& Mask);