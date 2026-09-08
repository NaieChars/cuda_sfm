#include "SFM.h"

// 两张图的SFM主流程
SFMResult runSFM(const CameraIntrinsics& intr)
{
    SFMResult result;
    
    //==================== SIFT ================
    std::vector<FeatureSet> features = extractFeaturesFromImages();
    if (features.size() < 2)    
    {
        std::cerr << "[SFM] Not enough images." << std::endl;
        return result;
    }

    // ================= Match ==================
    std::vector<cuda_MatchResult> matchResult = cuda_FeatureMatch(features[0], features[1]);

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    convertMatches(features[0], features[1], matchResult, points1, points2);

    // ================ Essential Matrix ================
    std::vector<uchar> essentialMask;
    cv::Mat E = estimateEssentialMatrix(points1, points2, intr, essentialMask);

    // ================= Pose Recovery =================
    cv::Mat R, t;
    int inlierCount = 0;
    std::vector<uchar> poseMask;

    bool poseOK = recoverCameraPose(E, points1, points2, intr, essentialMask, R, t, inlierCount, poseMask);
    if (!poseOK)
    {
        std::cerr << "[SFM] Error: Pose recovery failed." << std::endl;
        return result;
    }

    result.cameras.resize(2);
    result.cameras[0].R = cv::Mat::eye(3, 3, CV_64F);
    result.cameras[0].t = cv::Mat::zeros(3, 1, CV_64F);
    result.cameras[1].R = R;
    result.cameras[1].t = t;

    // ================ Triangulation ===============
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

    // --------- 调试代码：检查 Points3D 是否写对 ------------
    std::cout << "Points3D size = " << Points3D.size() << '\n';

    for (int i = 0; i < std::min(10, (int)Points3D.size()); ++i)
    {
        const auto& p = Points3D[i];
        std::cout << i << ": (" << p.x << ", "
                << p.y << ", " << p.z << ")\n";
    }
    // ----------------------------------------------------------

    return result;
}

