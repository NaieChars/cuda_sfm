#pragma once
#include <opencv2/opencv.hpp>
#include <random>

#include "CameraIntrinsics.h"

// ------------------ 外部接口 --------------------

struct RansacResult
{
    cv::Mat E;                      // 最终E
    std::vector<int> essentialMask; // 最终掩码
    int iterationsUsed = 0;         // 实际进行的迭代次数
};

// Opencv求E的顶层对外接口
cv::Mat estimateEssentialMatrix(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
        const CameraIntrinsics& intr,std::vector<uchar>& essentialMask);

// CUDA 版本求E的对外上层接口，对输出结果暂时不包装成 RansacResult
cv::Mat estimateEssentialMatrixRANSAC(const std::vector<cv::Point2f>& points1,
                        const std::vector<cv::Point2f>& points2,
                        double fx, double fy, double cx, double cy, 
                        const float confidence,
                        const int maxIterations,
                        std::vector<uchar>& essentialMask);


// -------------- 内部计算函数 -------------------
// 8组归一化匹配点，输出一个候选E
cv::Mat generateCandidateE(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2);

// 输入多个已经归一化的匹配点，重新拟合最终E
cv::Mat refineEssentialMatrix(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2);

// 对两个点序列（已经归一化）进行随机八点采样
void sample8Points(const std::vector<cv::Point2f>& points1, 
                   const std::vector<cv::Point2f>& points2,
                   std::vector<cv::Point2f>& sample1,
                   std::vector<cv::Point2f>& sample2,
                   std::mt19937& gen);