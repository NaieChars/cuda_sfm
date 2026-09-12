#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "CameraIntrinsics.h"
#include "DataStruct.h"
#include "Config.h"

/**
 * @brief 使用 CUDA 对匹配点进行三角化，重建三维点
 *
 * 根据两幅图像的相机位姿和内参构建投影矩阵，
 * 在 CUDA 上并行完成三角化，并通过深度和重投影误差过滤无效三维点。
 *
 * @param inlierPoints1 图像1中的有效匹配点，已经过两次筛选
 * @param inlierPoints2 图像2中的有效匹配点，与 inlierPoints1 一一对应
 * @param intr 相机内参
 * @param pose1 图像1对应的相机位姿
 * @param pose2 图像2对应的相机位姿
 * @param Mask3D 输出：每个输入点对应的三维点有效性掩码，1 表示有效，0 表示无效
 * @return 通过深度和重投影误差验证的有效三维点，已通过 Mask3D 过滤
 */
std::vector<cv::Point3f> cudaTriangulation(
    const std::vector<cv::Point2f>& inlierPoints1,
    const std::vector<cv::Point2f>& inlierPoints2,
    const CameraIntrinsics& intr,
    const CameraPose& pose1,
    const CameraPose& pose2, 
    std::vector<int>& Mask3D);