#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include "CameraIntrinsics.h"
#include "DataStruct.h"
#include "Config.h"

/**
 * @brief 使用 RANSAC-PnP 估计新视角的相机位姿
 *
 * 根据已有的 3D 点以及它们在当前图像中的 2D 投影点，
 * 调用 OpenCV 的 solvePnPRansac() 估计当前相机的旋转矩阵 R
 * 和平移向量 t，并通过 RANSAC 筛选有效的 3D-2D 对应关系。
 *
 * 输入的 objectPoints 和 imagePoints 必须一一对应：
 * objectPoints[i] 表示第 i 个 3D 点，
 * imagePoints[i] 表示该 3D 点在当前图像中的对应 2D 点。
 *
 * @param objectPoints 已有的 3D 点，作为 PnP 的输入
 * @param imagePoints 上述 3D 点在当前图像中的对应 2D 位置
 * @param intr 相机内参
 * @return PnPResult，包括估计得到的相机位姿、内点索引以及求解是否成功
 */
PnPResult solvePnPForNewView(
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints,
    const CameraIntrinsics& intr
);