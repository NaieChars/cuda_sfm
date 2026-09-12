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

// 一起过滤原始坐标与点坐标
void filterInliers(
    const std::vector<cv::Point2f>& points,
    const std::vector<int>& indices,
    const std::vector<uchar>& mask,
    std::vector<cv::Point2f>& outPoints,
    std::vector<int>& outIndices)
{
    for (size_t i = 0; i < mask.size(); i++)
    {
        if (mask[i])
        {
            outPoints.push_back(points[i]);
            outIndices.push_back(indices[i]);
        }
    }
}


// 匹配点的坐标序列进行相机坐标归一化
std::vector<cv::Point2f> normalizePoints(const std::vector<cv::Point2f>& points,
double fx, double fy, double cx, double cy)
{
    std::vector<cv::Point2f> normalizedPoints;
    normalizedPoints.reserve(points.size());

    for (const auto& p : points)
    {
        float x = static_cast<float>((p.x - cx) / fx);
        float y = static_cast<float>((p.y - cy) / fy);

        normalizedPoints.emplace_back(x, y);
    }
    return normalizedPoints;
}

// flatten Matrix(3 x 4)，传入 CV_64F
std::array<double, 12> flattenMatrix_3x4(const cv::Mat& P)
{
    CV_Assert(P.type() == CV_64F);
    CV_Assert(P.rows == 3 && P.cols == 4);

    std::array<double, 12> flat;

    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 4; c++)
            flat[r * 4 + c] = P.at<double>(r, c);
    }
    return flat;
}

// flatten Matrix(3 x 3)，传入 CV_64F
std::array<double, 9> flattenMatrix_3x3(const cv::Mat& E)
{
    CV_Assert(E.type() == CV_64F);
    CV_Assert(E.rows == 3 && E.cols == 3);

    std::array<double, 9> flat;

    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
            flat[r * 3 + c] = E.at<double>(r, c);
    }
    return flat;
}

// 将一个两位32位整数打包成一个64位key
long long makeTrackKey(int imageIdx, int kpIdx)
{
    return (static_cast<long long>(imageIdx) << 32) | static_cast<unsigned int>(kpIdx);
}

// 将 NewMatch 类型数据转换成两组像素坐标，PnP阶段用于将新点进行三角化
void convertNewMatches(
    const FeatureSet& featuresPrev,
    const FeatureSet& featuresCur,
    const std::vector<NewMatch>& newMatches,
    std::vector<cv::Point2f>& pointsPrev,
    std::vector<cv::Point2f>& pointsCur)
{
    for (const auto& match : newMatches)
    {
        pointsPrev.emplace_back(
            featuresPrev.kpX[match.PrevImagePointIdx],
            featuresPrev.kpY[match.PrevImagePointIdx]
        );

        pointsCur.emplace_back(
            featuresCur.kpX[match.CurImagePointIdx],
            featuresCur.kpY[match.CurImagePointIdx]
        );
    }
}
