#include "SFM.h"

// 两张图的SFM主流程
SFMResult runSFM(const CameraIntrinsics& intr)
{
    SFMResult result;
    
    //------------ SIFT ---------------
    std::vector<FeatureSet> features = extractFeaturesFromImages();
    if (features.size() < 2)    
    {
        std::cerr << "[SFM] Not enough images." << std::endl;
        return result;
    }
    
    std::cout << "SIFT start." << std::endl;

    // ---------- Match ----------------
    std::vector<cuda_MatchResult> matchResult = cuda_FeatureMatch(features[0], features[1]);

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    convertMatches(features[0], features[1], matchResult, points1, points2);

    std::cout << "Match start." << std::endl;

    // ---------- Essential Matrix ------------
    std::vector<uchar> essentialMask;
    cv::Mat E = estimateEssentialMatrix(points1, points2, intr, essentialMask);

    std::cout << "Find E start." << std::endl;

    // ------------- Pose Recovery -----------
    cv::Mat R, t;
    int inlierCount = 0;
    std::vector<uchar> poseMask;

    bool poseOK = recoverCameraPose(E, points1, points2, intr, essentialMask, R, t, inlierCount, poseMask);
    if (!poseOK)
    {
        std::cerr << "[SFM] Error: Pose recovery failed." << std::endl;
        return result;
    }

    std::cout << "PoseRecovary start." << std::endl;

    // ------------ Triangulation --------------
    // 这里显式处理一下特征点筛选
    std::vector<cv::Point2f> inlierPoints1 = fliterInlierPoints(points1, essentialMask);
    std::vector<cv::Point2f> inlierPoints2 = fliterInlierPoints(points2, essentialMask);
    std::vector<cv::Point2f> poseInlierPoints1 = fliterInlierPoints(inlierPoints1, poseMask);
    std::vector<cv::Point2f> poseInlierPoints2 = fliterInlierPoints(inlierPoints2, poseMask);

    std::vector<int> Mask3D;

    std::vector<cv::Point3f> Points3D = cudaTriangulation(poseInlierPoints1, poseInlierPoints2, intr, R, t, Mask3D);

    for (auto p : Points3D)
    {
        ColoredPoint3D points;

        points.r = 255;
        points.g = 255;
        points.b = 255;

        result.pointCloud.push_back(points);
    }

    std::cout << "Triangulation start." << std::endl;

    return result;
}

