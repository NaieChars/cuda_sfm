#include "PnP.h"

// 使用已有3D点 + 他们对于当前图像中对应的2D点，求新相机在世界坐标系下的位姿
PnPResult solvePnPForNewView(
    const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints,
    const CameraIntrinsics& intr)
{
    PnPResult result;

    if (objectPoints.size() < 4 || objectPoints.size() != imagePoints.size())
    {
        return result;
    }

    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    cv::Mat distCoeffs; // 空 Mat，暂无畸变系数
    const int iterationsCount = 200;
    float reprojectionError = 2.5;
    float confidence = 0.999;


    bool OK = cv::solvePnPRansac(
        objectPoints,
        imagePoints,
        intr.K,
        distCoeffs,
        rvec,
        tvec,
        true,
        iterationsCount,
        reprojectionError,
        confidence,
        inliers,
        cv::SOLVEPNP_ITERATIVE
    );

    if (!OK)
        return result;
    
    // rvec -> R
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    result.R = R;
    result.t = tvec;
    result.inlierIndiecs = inliers;
    result.success = true;

    std::cout << "[PnPSolver] PnP solved successfully. Inliers: " 
    << inliers.size() << " / " << objectPoints.size() << std::endl;

    return result;
}