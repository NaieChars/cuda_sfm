#include "PnP.h"


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
        false,
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

    // ------------- 调试代码 -------------
    #if DEBUG_OUTPUT
        std::cout << "PnP t:\n" << tvec << "\n";
    #endif

    result.pose.R = R;
    result.pose.t = tvec;
    result.inlierIndices = inliers;
    result.success = true;

    std::cout << "[PnPSolver] PnP solved successfully. Inliers: " 
    << inliers.size() << " / " << objectPoints.size() << std::endl;

    return result;
}