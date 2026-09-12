#pragma once

struct cuda_FeatureSet
{
	int numFeatures;
	float* descriptors = nullptr;
};


/**
 * CUDA 特征匹配结果
 *
 * 成员：
 * - bestIdx  当前图像特征点在第二张图像中的最佳匹配点索引
 * - bestDist 当前特征点与最佳匹配点之间的距离
 */
struct cuda_MatchResult
{
    int bestIdx;    ///< 图1特征点对应的图2特征点的 SIFT 索引
    float bestDist; ///< 两个特征点描述子之间的最佳距离
};