#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "FeatureSet.h"


// SIFT 主流程
std::vector<FeatureSet> extractFeaturesFromImages();