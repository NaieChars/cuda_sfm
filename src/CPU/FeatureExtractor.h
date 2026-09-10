#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "DataStruct.h"


// SIFT 主流程
std::vector<FeatureSet> extractFeaturesFromImages(std::vector<cv::Mat>& images);