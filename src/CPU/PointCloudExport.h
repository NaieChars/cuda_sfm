#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "DataStruct.h"

struct ColoredPoint3D 
{
    cv::Point3f position;
    unsigned char r, g, b;
};

// 保存3D点云为ply文件
bool savePointCloudPLY(const std::vector<ColoredPoint3D>& points, const std::string& path);

// 提取点云颜色
ColoredPoint3D getColorPoint3D(int pointIndex, int imagePointIndex, cv::Mat& iamge,
    std::vector<cv::Point2f>& poseInlierPoints1, std::vector<cv::Point3f>& Points3D
    );