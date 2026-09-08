#pragma once
#include <vector>
#include "../GPU/d_FeatureSet.cuh"
#include "FeatureSet.h"

std::vector<cuda_MatchResult> cuda_FeatureMatch(const FeatureSet& f1, const FeatureSet& f2);

void convertMatches(const FeatureSet& f1, const FeatureSet& f2, const std::vector<cuda_MatchResult>& cuda_result,
                    std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);