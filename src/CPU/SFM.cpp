#include "SFM.h"
#include "Config.h"

// 增量式SFM主流程
SFMResult runSFM(const CameraIntrinsics& intr, int& imagesProcessed)
{
    SFMResult result;
    std::unordered_map<long long, int> track;
    imagesProcessed = 0;
    std::vector<cv::Mat> images;
    
    //==================== SIFT ================
    std::vector<FeatureSet> features = extractFeaturesFromImages(images);
    if (features.size() < 2)    
    {
        std::cerr << "[SFM] Not enough images." << std::endl;
        return result;
    }

    const int pictureNum = static_cast<int>(images.size());
    auto& pointCloud = result.pointCloud;
    auto& observations = result.observations;
    result.cameras.resize(pictureNum);
    auto& cameras = result.cameras;

    // ================= Match ==================
    std::vector<cuda_MatchResult> matchResult = cuda_FeatureMatch(features[0], features[1]);

    int count = countValidFeature(matchResult);
    std::cout << "[SFM] img0 - img1: " << count << " matches" << std::endl;
    
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    std::vector<int> indices1;
    std::vector<int> indices2;

    convertMatches(features[0], features[1], matchResult, points1, points2, indices1, indices2);

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

    result.cameras[0].R = cv::Mat::eye(3, 3, CV_64F);
    result.cameras[0].t = cv::Mat::zeros(3, 1, CV_64F);
    result.cameras[1].R = R;
    result.cameras[1].t = t;

    // ================ Triangulation ===============
    // 这里显式处理一下特征点与SIFT原始点筛选
    std::vector<cv::Point2f> inlierPoints1;
    std::vector<cv::Point2f> inlierPoints2;
    std::vector<int> inlierIndices1;
    std::vector<int> inlierIndices2;

    filterInliers(points1, indices1, essentialMask, inlierPoints1, inlierIndices1);
    filterInliers(points2, indices2, essentialMask, inlierPoints2, inlierIndices2);

    std::vector<cv::Point2f> poseInlierPoints1;
    std::vector<cv::Point2f> poseInlierPoints2;
    std::vector<int> poseInlierIndices1;
    std::vector<int> poseInlierIndices2;

    filterInliers(inlierPoints1, inlierIndices1, poseMask, poseInlierPoints1, poseInlierIndices1);
    filterInliers(inlierPoints2, inlierIndices2, poseMask, poseInlierPoints2, poseInlierIndices2);

    std::vector<int> Mask3D;

    std::vector<cv::Point3f> Points3D = cudaTriangulation(poseInlierPoints1, poseInlierPoints2, intr, cameras[0], cameras[1], Mask3D);

    int pointIndex = 0;
    for (int i = 0; i < Mask3D.size(); i++)
    {
        if (Mask3D[i] == 0) continue;

        ColoredPoint3D color3D = getColorPoint3D(pointIndex,i, images[0], poseInlierPoints1, Points3D);
        int landmarkIdx = static_cast<int>(pointCloud.size()); // landmarkIdx 是 3D 点在pointCloud中的编号
        pointCloud.push_back(color3D);

        int kp0 = poseInlierIndices1[i];
        int kp1 = poseInlierIndices2[i];

        track[makeTrackKey(0, kp0)] = landmarkIdx;
        track[makeTrackKey(1, kp1)] = landmarkIdx;

        observations.push_back({0, landmarkIdx, cv::Point2f(poseInlierPoints1[i].x, poseInlierPoints1[i].y)});
        observations.push_back({1, landmarkIdx, cv::Point2f(poseInlierPoints2[i].x, poseInlierPoints2[i].y)});

        pointIndex++;
    }

    imagesProcessed = 2;

    // =================== 链式扩展至 N 张图 =====================
    int lastRegisteredIdx = 1;

    for (int i = 2; i < pictureNum; i++)
    {
        int prevIdx = lastRegisteredIdx;

        std::vector<cuda_MatchResult> matchesPrev = cuda_FeatureMatch(features[prevIdx], features[i]);
        int count = countValidFeature(matchResult);
        std::cout << "\nimg" << prevIdx << "-img" << i << " matched points: " << count << std::endl;

        std::vector<cv::Point3f> pnpObjectPoints;   // 已有3D点
        std::vector<cv::Point2f> pnpImagePoints;    // 这些3D点在当前图像中的2D位置
        std::vector<NewMatch> newMatches;
        std::vector<PnPCorrespondence> pnpCorrespondences;

        assert(matchesPrev.size() == features[prevIdx].numFeatures);
        for (int kpPrev = 0; kpPrev < features[prevIdx].numFeatures; kpPrev++)
        {
            int kpCur = matchesPrev[kpPrev].bestIdx;

            if (kpCur < 0) continue;

            auto it = track.find(makeTrackKey(prevIdx, kpPrev));
            if (it != track.end())
            {
                int landmarkIdx = it->second;   // landmarkIdx：该特征点在pointcloud里面对应的下标
                pnpObjectPoints.push_back(pointCloud[landmarkIdx].position);
                pnpImagePoints.emplace_back(features[i].kpX[kpCur], features[i].kpY[kpCur]);
                pnpCorrespondences.push_back({landmarkIdx, kpCur});
            }
            else
            {
                newMatches.push_back({kpPrev, kpCur});
            }
        }
        
        std::cout << "Available 3D-2D points: " << pnpObjectPoints.size()
        << ", new points: " << newMatches.size() << std::endl;

        PnPResult pnpResult = solvePnPForNewView(pnpObjectPoints, pnpImagePoints, intr);
        if (!pnpResult.success) // 若 PnP 没成功
        {
            std::cerr << "[SFM] img" << i << " PnP failed. Skip it." << std::endl;
            continue;
        }

        cameras[i].R = pnpResult.R;
        cameras[i].t = pnpResult.t;
        imagesProcessed++;
        lastRegisteredIdx = i;

        for (int idx : pnpResult.inlierIndiecs)
        {
            int landmarkIdx = pnpCorrespondences[idx].landmarkIdx;
            int kpCur = pnpCorrespondences[idx].kpCur;

            track[makeTrackKey(i, kpCur)] = landmarkIdx;

            cv::Point2f uv(features[i].kpX[kpCur], features[i].kpY[kpCur]);
            observations.push_back({i, landmarkIdx, cv::Point2f(uv.x, uv.y)});
        }

        // 准备新匹配的2D点
        std::vector<cv::Point2f> newPointsPrev;
        std::vector<cv::Point2f> newPointsCur;

        convertNewMatches(features[prevIdx], features[i], newMatches, newPointsPrev, newPointsCur);

        std::vector<int> Mask3D;
        std::vector<cv::Point3f> Points3D =
        cudaTriangulation(newPointsPrev, newPointsCur, intr, cameras[prevIdx], cameras[i], Mask3D);
        
        // --------------- 调试输出 ---------------
        std::cout << "Mask3D: " << Mask3D.size() << std::endl;
        std::cout << "Points3D: " << Points3D.size() << std::endl;
        std::cout << "newPointsPrev: " << newPointsPrev.size() << std::endl;
        std::cout << "newPointsCur: " << newPointsCur.size() << std::endl;

        int pointIndex = 0;
        for (int k = 0; k < Mask3D.size(); k++)
        {
            if (Mask3D[k] == 0) continue;

            int kpPrev = newMatches[k].kpPrev;
            int kpCur  = newMatches[k].kpCur;

            ColoredPoint3D point = getColorPoint3D(pointIndex, k, images[prevIdx], newPointsPrev, Points3D);
            int landmarkIdx = static_cast<int>(pointCloud.size());
            pointCloud.push_back(point);

            track[makeTrackKey(prevIdx, kpPrev)] = landmarkIdx;
            track[makeTrackKey(i, kpCur)] = landmarkIdx;

            observations.push_back({prevIdx, landmarkIdx, cv::Point2f(newPointsPrev[k].x, newPointsPrev[k].y)});
            observations.push_back({i, landmarkIdx, cv::Point2f(newPointsCur[k].x, newPointsCur[k].y)});

            pointIndex++;
        }
    }

    // =================== 保存并生成.ply文件 ========================
    if (savePointCloudPLY(pointCloud, "3D_cloud.ply"))
    {
        std::cout << "[SavePLY] Successfully!" << std::endl;
    }
    else
    {
        std::cerr << "[SavePLY] Failed!" << std::endl;
    }

    return result;
}

