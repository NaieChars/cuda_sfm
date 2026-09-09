#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// 根据求掩码 Mask 从特征点中进行筛选
std::vector<cv::Point2f>fliterInlierPoints(const std::vector<cv::Point2f>& points, const std::vector<uchar>& Mask);

// 匹配点的坐标序列进行相机坐标归一化
std::vector<cv::Point2f> normalizePoints(const std::vector<cv::Point2f>& points,
double fx, double fy, double cx, double cy);

std::array<double, 12> flattenMatrix_3x4(const cv::Mat& P);

std::array<double, 9> flattenMatrix_3x3(const cv::Mat& E);