#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


/**
 * @brief 带颜色的三维点
 *
 * 用于保存 SFM 最终重建得到的三维点及其对应的 RGB 颜色，
 * 最终写入 .ply 点云文件。
 *
 * 包含：
 * 
 * - 三维点在世界坐标系中的位置
 * 
 * - 该点对应的 RGB 颜色
 */
struct ColoredPoint3D
{
    cv::Point3f position; ///< 三维点在世界坐标系中的位置
    unsigned char r, g, b; ///< 三维点对应的 RGB 颜色
};


// 保存3D点云为ply文件
bool savePointCloudPLY(const std::vector<ColoredPoint3D>& points, const std::string& path);



/**
 * @brief 根据三维点及其对应的图像点获取带颜色的三维点

    将 Points3D 中的三维点与 poseInlierPoints 中对应的图像坐标关联，
    从原始图像中读取该像素的颜色，并生成 ColoredPoint3D。


    注意：
    pointIndexInPoints3D 和 pointIndexInPoseInliers 分别对应过滤后的
    三维点和过滤前的图像内点，因此两个索引一般情况下不同。


    @param pointIndexInPoints3D Points3D 中三维点的索引
    @param pointIndexInPoseInliers poseInlierPoints 中对应图像点的索引
    @param image 原始图像
    @param poseInlierPoints 图像中的位姿内点坐标
    @param Points3D 经过 Mask3D 过滤后的有效三维点
    @return 包含三维坐标和图像颜色的 ColoredPoint3D
 */
ColoredPoint3D getColorPoint3D(
    int pointIndexInPoints3D,
    int pointIndexInPoseInliers,
    cv::Mat& image,
    const std::vector<cv::Point2f>& points,
    const std::vector<cv::Point3f>& Points3D);