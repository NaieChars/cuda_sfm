#pragma once

#include <opencv2/opencv.hpp>

#include "CameraIntrinsics.h"
#include "util.h"
#include "Config.h"


/**
 * @brief 从本质矩阵恢复两幅图像之间的相对相机位姿
 *
 * 首先根据 essentialMask 筛选 E 的内点匹配，
 * 然后调用 OpenCV 的 recoverPose() 从本质矩阵 E 中恢复旋转矩阵 R
 * 和平移方向 t，并通过三角化检查进一步确定位姿内点。
 *
 * 两级 Mask 的含义：
 * 
 *   essentialMask：RANSAC 阶段得到的匹配点内点掩码，
 *                  用于从 points1/points2 中筛选输入点。
 *
 *   poseMask：recoverPose() 根据恢复的相机位姿得到的内点掩码，
 *             对应筛选后的 inlierPoints1/inlierPoints2。
 *
 * @param E 本质矩阵
 * @param points1 图像1中的匹配点
 * @param points2 图像2中的匹配点
 * @param intr 相机内参
 * @param essentialMask 本质矩阵 RANSAC 得到的内点掩码
 * @param R 输出：图2相对于图1的旋转矩阵
 * @param t 输出：图2相对于图1的平移方向
 * @param inlierCount 输出：recoverPose 最终得到的内点数量
 * @param poseMask 输出：recoverPose 得到的位姿内点掩码
 * @return true 表示位姿恢复成功，false 表示有效内点过少
 */
bool recoverCameraPose(
    const cv::Mat3d& E, 
    const std::vector<cv::Point2f>& points1, 
    const std::vector<cv::Point2f>& points2,
    const CameraIntrinsics& intr,
    const std::vector<uchar>& essentialMask,
    cv::Mat& R, cv::Mat& t, int& inlierCount, 
    std::vector<uchar>& poseMask);
