#include "SFM.h"
#include "Config.h"

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

    #if USE_CUDA_RANSAC
    cv::Mat E = estimateEssentialMatrixRANSAC(points1, points2, intr.fx, intr.fy, intr.cx, intr.cy, 0.999, 1000, essentialMask);
    std::cout << "[CUDA RANSAC] E:\n" << E << std::endl;
    #else
    cv::Mat E = estimateEssentialMatrix(points1, points2, intr, essentialMask);
    std::cout << "[OpenCV] E:\n" << E << std::endl;
    #endif

    std::cout << "Inliers: " << cv::countNonZero(essentialMask) << " / " << points1.size() << std::endl;

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

    int pointIndex = 0;
    for (int i = 0; i < Mask3D.size(); i++)
    {
        if (Mask3D[i] == 0) continue;

        ColoredPoint3D color3D = getColorPoint3D(pointIndex, i, features[0], poseInlierPoints1, Points3D);
        result.pointCloud.push_back(color3D);
        pointIndex++;
    }

    // =================== 保存并生成.ply文件 ========================
    if (savePointCloudPLY(result.pointCloud, "3D_cloud.ply"))
    {
        std::cout << "[SavePLY] Successfully!" << std::endl;
    }
    else
    {
        std::cerr << "[SavePLY] Failed!" << std::endl;
    }

    return result;
}

