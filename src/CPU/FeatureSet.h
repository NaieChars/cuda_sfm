#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>

struct FeatureSet
{
	static constexpr int DESC_DIM = 128;

	int numFeatures = 0;

	std::vector<float> kpX;	
	std::vector<float> kpY;
	std::vector<float> descriptors;	

	std::vector<cv::KeyPoint> cvKeypoints;

	cv::Mat image;	// 保存原图，后面增量式设计更高层数据结构，Image 与 feature 同级
};

