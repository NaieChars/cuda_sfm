#pragma once

#include "CameraIntrinsics.h"
#include "util.h"
#include "cuda_Check.cuh"
#include "../GPU/findEssentialMat.cuh"

#include <cuda_runtime.h>
#include <set>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <random>
/**
 * @brief RANSAC 本质矩阵估计结果
 *
 * 包含
 * 
 * - E 最终估计的本质矩阵
 * 
 * - essentialMask 内点掩码
 * 
 * - iterationsUsed 实际迭代次数
 * 
 */
struct RansacResult
{
    cv::Mat E;                      ///< 最终估计得到的本质矩阵
    std::vector<uchar> essentialMask; ///< 内点掩码，1 表示内点，0 表示外点
    int iterationsUsed = 0;         ///< RANSAC 实际执行的迭代次数
};



// ------------------ 外部接口 --------------------

/**
 * @brief 使用 OpenCV API 估计本质矩阵
 *
 * 调用 cv::findEssentialMat()，使用 RANSAC 方法从两张图像的匹配点中
 * 估计本质矩阵 E，并生成对应的内点掩码。
 *
 * @param points1 图像1中的匹配点
 * @param points2 图像2中的匹配点
 * @param intr 相机内参
 * @return RANSAC 本质矩阵估计结果，包括 E 和内点掩码
 */
RansacResult estimateEssentialMatrix(
    const std::vector<cv::Point2f>& points1, 
    const std::vector<cv::Point2f>& points2,
    const CameraIntrinsics& intr);



/**
 * @brief 使用 CUDA 加速 RANSAC 估计本质矩阵
 *
 * 使用归一化后的匹配点，通过 RANSAC 随机采样 8 个点生成候选本质矩阵，
 * 再利用 CUDA 并行计算所有匹配点的 Sampson 误差并统计内点数量。
 *
 *
 * CPU 主要负责随机采样和候选 E 的生成，
 * CUDA 主要负责大量匹配点的并行误差计算和内点统计。
 *
 * @param points1 图像1中的匹配点
 * @param points2 图像2中的匹配点
 * @param intr 相机内参
 * @param confidence RANSAC 置信度，用于计算所需迭代次数
 * @param maxIterations 最大 RANSAC 迭代次数
 * @return RANSAC 本质矩阵估计结果，包括最终 E、内点掩码和实际迭代次数
 */
RansacResult estimateEssentialMatrixRANSAC(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const CameraIntrinsics& intr,
    const float confidence,
    const int maxIterations);


// ---------------- 内部计算函数 -------------------

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