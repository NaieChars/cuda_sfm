#include "util.h"

// 根据求掩码 Mask 从特征点中进行筛选
std::vector<cv::Point2f>fliterInlierPoints(const std::vector<cv::Point2f>& points, const std::vector<uchar>& Mask)
{
    std::vector<cv::Point2f> good_points;
    for (size_t i = 0; i < Mask.size(); i++)
    {
        if (Mask[i])
        {
            good_points.push_back(points[i]);
        }
    }
    return good_points;
}