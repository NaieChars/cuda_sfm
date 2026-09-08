#include "EssentialMatrix.h"

cv::Mat estimateEssentialMatrix(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
        const CameraIntrinsics& intr,std::vector<uchar>& essentialMask)
{
    cv::Mat E = cv::findEssentialMat(
        points1, points2,
        intr.K,
        cv::RANSAC,
        0.999,
        1.5,
        essentialMask
    );

    return E;
}