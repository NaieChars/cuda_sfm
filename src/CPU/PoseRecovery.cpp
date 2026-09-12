#include "PoseRecovery.h"


bool recoverCameraPose(
    const cv::Mat3d& E, 
    const std::vector<cv::Point2f>& points1, 
    const std::vector<cv::Point2f>& points2,
    const CameraIntrinsics& intr,
    const std::vector<uchar>& essentialMask,
    cv::Mat& R, cv::Mat& t, int& inlierCount, 
    std::vector<uchar>& poseMask)
{
    std::vector<cv::Point2f> inlierPoints1, inlierPoints2;

    inlierPoints1 = fliterInlierPoints(points1, essentialMask);
    inlierPoints2 = fliterInlierPoints(points2, essentialMask);

    inlierCount = cv::recoverPose(E, inlierPoints1, inlierPoints2, intr.K, R, t, poseMask);
    if (inlierCount <= 15)
    {
        std::cerr << "[PoseRecovery] Final inliers too low, discard." << std::endl;
        return false;
    }

    #if DEBUG_OUTPUT
        std::cout << "[PoseRecovery] R:\n" << R << std::endl;
        std::cout << "t:\n" << t << std::endl;
    #endif

    std::cout << "[PoseRecovery] Final inliers: " << inlierCount << std::endl;
    return true;
}

