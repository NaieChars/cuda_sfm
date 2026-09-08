#pragma once

#include <opencv2/opencv.hpp>
#include <array>

struct CameraIntrinsics
{
	float fx, fy, cx, cy;
	cv::Mat K;  // 3 x 3 内参矩阵 K
	cv::Mat Kinv;   // K 逆
    // 暂无畸变
};


CameraIntrinsics createCameraIntrinsics(float fx, float fy, float cx, float cy);

// 把像素坐标(px,py)转换成归一化相机坐标(不考虑畸变，纯针孔模型)
cv::Vec2f pixelToNormalized(const CameraIntrinsics& intr, float px, float py);
