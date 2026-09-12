#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

#include "DataStruct.h"
#include "SFMTypes.h"

// 根据求掩码 Mask 从特征点中进行筛选
std::vector<cv::Point2f>fliterInlierPoints(const std::vector<cv::Point2f>& points, const std::vector<uchar>& Mask);

// 根据掩码筛选并保存原Index
void filterInliers(
    const std::vector<cv::Point2f>& points,
    const std::vector<int>& indices,
    const std::vector<uchar>& mask,
    std::vector<cv::Point2f>& outPoints,
    std::vector<int>& outIndices);

// 匹配点的坐标序列进行相机坐标归一化
std::vector<cv::Point2f> normalizePoints(const std::vector<cv::Point2f>& points,
                    double fx, double fy, double cx, double cy);

std::array<double, 12> flattenMatrix_3x4(const cv::Mat& P);

std::array<double, 9> flattenMatrix_3x3(const cv::Mat& E);

long long makeTrackKey(int imageIdx, int kpIdx);

void convertNewMatches(
    const FeatureSet& featuresPrev,
    const FeatureSet& featuresCur,
    const std::vector<NewMatch>& newMatches,
    std::vector<cv::Point2f>& pointsPrev,
    std::vector<cv::Point2f>& pointsCur);
