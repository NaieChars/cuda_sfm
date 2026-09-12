#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "DataStruct.h"



/**
 * @brief SIFT 特征提取阶段的对外接口
 *
 * 负责读取所有输入图像，并为每张图像提取 SIFT 特征。
 *
 * 内部流程： 
 *  
 * - 调用 robustImreadIndexed() 按索引读取图像，直到无法读取为止；
 * 
 * - 将读取到的图像保存到 images 中；
 * 
 * - 对每张图像调用 extractSIFTFeatures() 提取 SIFT 特征；
 * 
 * - 将每张图像对应的 FeatureSet 保存到结果容器中返回。
 *
 * @param images 输出：存放读取到的所有图像矩阵
 * @return 每张图像对应的 SIFT 特征集合，按照图像索引顺序排列
 */
std::vector<FeatureSet> extractFeaturesFromImages(std::vector<cv::Mat>& images);