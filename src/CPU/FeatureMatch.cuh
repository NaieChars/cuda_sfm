#pragma once
#include <vector>
#include "../GPU/d_FeatureSet.cuh"
#include "DataStruct.h"

/**
 * @brief 统计有效匹配点
 * @param h_resutl CUDA特征匹配结果
 * @return 有效匹配点对数
 */
int countValidFeature(const std::vector<cuda_MatchResult>& h_result);

/**
 * @brief 特征点匹配对外接口
 * 
 * 将CPU端的描述子传入GPU，进行暴力匹配得到最佳匹配点，GPU传回对应特征点的SIFT索引
 * 
 * @param f1 第一张图的 SIFT 数据集合
 * @param f2 第二张图的 SIFT 数据集合
 * 
 * @return CUDA 匹配结果数据集合
 */
std::vector<cuda_MatchResult> cuda_FeatureMatch(const FeatureSet& f1, const FeatureSet& f2);

/**
 * @brief 将 CUDA 特征匹配结果转换为 OpenCV 友好的点对格式
 *
 * 遍历 CUDA 匹配结果，将有效匹配转换为两张图像中对应的
 * cv::Point2f 坐标，并同时保存匹配点在原始 FeatureSet 中的下标。
 *
 * 四个输出 vector 保持严格的一一对应关系：即第 k 个元素共同表示同一对匹配点 
 * 
 * - points1[k] 图像 1 中的坐标
 * 
 * - points2[k] 图像 2 中的坐标
 * 
 * - indices1[k] 该点在 f1 中的原始特征下标
 * 
 * - indices2[k] 该点在 f2 中的原始特征下标
 *
 * @param f1 第一张图像的特征集合
 * @param f2 第二张图像的特征集合
 * @param cuda_result CUDA 特征匹配结果
 * @param points1 输出：第一张图像中的匹配点坐标
 * @param points2 输出：第二张图像中的匹配点坐标
 * @param indices1 输出：points1 对应的原始特征下标
 * @param indices2 输出：points2 对应的原始特征下标
 */
void convertMatches(const FeatureSet& f1, const FeatureSet& f2, const std::vector<cuda_MatchResult>& cuda_result,
                    std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2,
                    std::vector<int>& indices1, std::vector<int>& indices2);