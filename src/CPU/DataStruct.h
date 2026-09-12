#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

#include "PointCloudExport.h"



/**
 * @brief SIFT 特征集合
 *
 * 保存一张图像中提取到的 SIFT 特征，包括关键点坐标和描述子。
 *
 * kpX、kpY 和 descriptors 按相同的特征点索引对应：
 * 
 * 第 i 个特征点的坐标为 (kpX[i], kpY[i])，
 * 
 * 对应的描述子为 descriptors[i * DESC_DIM ... (i + 1) * DESC_DIM - 1]。
 *
 * cvKeypoints 仅用于将自定义 SIFT 数据转换为 OpenCV 格式的点坐标，
 * 主要供 convertMatches() 获取匹配点坐标。
 */
struct FeatureSet
{
    static constexpr int DESC_DIM = 128; ///< SIFT 描述子的维度

    int numFeatures = 0; ///< 特征点数量

    std::vector<float> kpX; ///< 所有关键点的 X 坐标
    std::vector<float> kpY; ///< 所有关键点的 Y 坐标

    ///< 所有 SIFT 描述子，每个特征点对应 DESC_DIM 个 float
    std::vector<float> descriptors;

    ///< OpenCV 格式的关键点，仅用于 convertMatches() 获取关键点坐标
    std::vector<cv::KeyPoint> cvKeypoints;
};



/**
 * @brief PnP 中的 3D-2D 对应关系
 *
 * 保存一个已有三维点与其在当前图像中对应的二维特征点之间的索引关系。
 *
 * landmarkIdx 对应 pointCloud 中的三维点，
 * kpCur 对应当前图像 features 中的二维特征点。
 */
struct PnPCorrespondence
{
    int landmarkIdx;            ///< 3D 点在 pointCloud 中的下标，从 track 的 value 中获取
    int CurImagePointIdx;       ///< 该 3D 点在当前图像中对应的 2D 特征点下标
};



/**
 * @brief 两张图像之间的一对特征匹配
 *
 * 保存同一对匹配点分别在前一张图像和当前图像中的特征点下标。
 */
struct NewMatch
{
    int PrevImagePointIdx; ///< 匹配点在前一张图像中的特征点下标
    int CurImagePointIdx;  ///< 匹配点在当前图像中的特征点下标
};



/**
 * @brief 相机位姿
 *
 * 描述世界坐标系中的三维点如何变换到当前相机坐标系：
 *
 *     X_camera = R * X_world + t
 */
struct CameraPose
{
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);   ///< 世界坐标到相机坐标的旋转矩阵
    cv::Mat t = cv::Mat::zeros(3, 1, CV_64F); ///< 世界坐标到相机坐标的平移向量
};



/**
 * @brief 一次三维点观测
 *
 * 表示某个相机对某个三维点的一次观测。
 * 一个 Observation 将相机、三维点以及该点在相机图像中的观测坐标关联起来。
 *
 * 包含
 * 
 * - cameraIdx 当前相机在相机数组的下标
 * 
 * - pointIdx 点在 pointCloud 中的下标
 * 
 * - uv 为该三维点在对应图像中的像素坐标
 */
struct Observation
{
    int cameraIdx;      ///< 观测该三维点的相机在相机数组中的下标
    int pointIdx;       ///< 被观测的三维点在 pointCloud 中的下标
    cv::Point2f uv;     ///< 该三维点在对应图像中的像素坐标
};



/**
 * @brief SFM 三维重建结果
 *
 * 保存 SFM 重建阶段得到的相机位姿、三维点云以及点与相机之间的观测关系。
 *
 * cameras、pointCloud 和 observations 分别用于描述相机、三维点以及
 * 三维点在各个相机中的观测信息，其中 observations 主要供 BA 使用。
 */
struct SFMResult
{
    std::vector<CameraPose> cameras;        ///< 每个视角的相机位姿，相机0固定为单位位姿
    std::vector<ColoredPoint3D> pointCloud; ///< 重建得到的带颜色三维点云
    std::vector<Observation> observations;  ///< 所有三维点观测记录，供 BA 使用
};



/**
 * @brief PnP 位姿估计结果
 *
 * 保存 PnP 算法估计得到的相机位姿，以及参与计算的匹配点中
 * 被判定为内点的点索引。
 *
 * 包含
 * 
 * - pose 保存 PnP 求得的旋转矩阵 R 和平移向量 t，
 * 
 * - inlierIndices 对应输入 objectPoints / imagePoints 中的原始下标。
 */
struct PnPResult
{
    CameraPose pose;                 ///< PnP 估计得到的相机位姿
    std::vector<int> inlierIndices;  ///< PnP 判断为内点的输入点下标
    bool success = false;            ///< PnP 是否成功
};