#include "SFM.h"
#include "Config.h"

// 增量式SFM主流程
SFMResult runSFM(const CameraIntrinsics& intr, int& imagesProcessed)
{
    SFMResult result;
    std::unordered_map<long long, int> track;
    std::vector<cv::Mat> images;
    imagesProcessed = 0;
    
    //==================== SIFT ================
    std::vector<FeatureSet> features = extractFeaturesFromImages(images);

    const int pictureNum = static_cast<int>(images.size());

    // std::cout << "[SFM] Found " << pictureNum << " images" << std::endl;
    if (pictureNum < 2)    
    {
        // std::cerr << "[SFM] Not enough images. You need to have at least 2 images." << std::endl;
        return result;
    }

    auto& pointCloud = result.pointCloud;
    auto& observations = result.observations;
    result.cameras.resize(pictureNum);
    auto& cameras = result.cameras;

    // ================= Match ==================
    std::vector<cuda_MatchResult> matchResult = cuda_FeatureMatch(features[0], features[1]);

    int count = countValidFeature(matchResult);
    std::cout << "[Match] img0 - img1: " << count << " matches" << std::endl;
    
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    std::vector<int> indices1;
    std::vector<int> indices2;

    convertMatches(features[0], features[1], matchResult, points1, points2, indices1, indices2);

    // ================ Essential Matrix ================
    RansacResult Eresult;

    #if USE_CUDA_RANSAC
        Eresult = estimateEssentialMatrixRANSAC(points1, points2, intr, 0.999, 1000);
        #if DEBUG_OUTPUT
            std::cout << "[CUDA RANSAC] E:\n" << Eresult.E << std::endl;
        #endif
    #else
        Eresult = estimateEssentialMatrix(points1, points2, intr);
        #if DEBUG_OUTPUT
            std::cout << "[OpenCV] E:\n" << Eresult.E << std::endl;
        #endif
    #endif


    // ================= Pose Recovery =================
    cv::Mat R, t;
    int inlierCount = 0;
    std::vector<uchar> poseMask;

    bool poseOK = recoverCameraPose(Eresult.E, points1, points2, intr, Eresult.essentialMask, R, t, inlierCount, poseMask);
    if (!poseOK)
    {
        //std::cerr << "[SFM] Error: Pose recovery failed." << std::endl;
        return result;
    }

    // 首次维护相机位姿
    result.cameras[0].R = cv::Mat::eye(3, 3, CV_64F);
    result.cameras[0].t = cv::Mat::zeros(3, 1, CV_64F);
    result.cameras[1].R = R;
    result.cameras[1].t = t;

    // ================ Triangulation ===============
    // 显式处理一下特征点与SIFT原始点筛选
    std::vector<cv::Point2f> inlierPoints1;
    std::vector<cv::Point2f> inlierPoints2;
    std::vector<int> inlierIndices1;
    std::vector<int> inlierIndices2;

    filterInliers(points1, indices1, Eresult.essentialMask, inlierPoints1, inlierIndices1);
    filterInliers(points2, indices2, Eresult.essentialMask, inlierPoints2, inlierIndices2);

    std::vector<cv::Point2f> poseInlierPoints1;  // 三角化用的序列
    std::vector<cv::Point2f> poseInlierPoints2;
    std::vector<int> poseInlierIndices1;    
    std::vector<int> poseInlierIndices2;

    filterInliers(inlierPoints1, inlierIndices1, poseMask, poseInlierPoints1, poseInlierIndices1);
    filterInliers(inlierPoints2, inlierIndices2, poseMask, poseInlierPoints2, poseInlierIndices2);

    // Mask3D 长度等于 poseInlierIndices1 长度
    std::vector<int> Mask3D;

    // Points3D 全是有效3D点，已经经过 Mask3D 过滤
    std::vector<cv::Point3f> Points3D = cudaTriangulation(poseInlierPoints1, poseInlierPoints2, intr, cameras[0], cameras[1], Mask3D);

    int pointIndexInPoints3D = 0; // 记录 Mask3D 里有效3D点的索引
    // 处理所有有效3D点
    // pointIdxInPoseInliers 是 Mask3D 中当前正在检查的原始输入点下标
    for (int pointIdxInPoseInliers = 0; pointIdxInPoseInliers < Mask3D.size(); pointIdxInPoseInliers++)
    {
        if (Mask3D[pointIdxInPoseInliers] == 0) continue;

        ColoredPoint3D color3D = getColorPoint3D(pointIndexInPoints3D, pointIdxInPoseInliers, images[0], poseInlierPoints1, Points3D);

        // landmarkIdx 是 3D 点在pointCloud中的编号，pointClouds 是包含这次重建所有3D点的容器
        int landmarkIdx = static_cast<int>(pointCloud.size()); 
        pointCloud.push_back(color3D);

        int kp0 = poseInlierIndices1[pointIdxInPoseInliers];    // 当前3D点在原始SIFT中的对应下标
        int kp1 = poseInlierIndices2[pointIdxInPoseInliers];

        // 首次构建 track
        track[makeTrackKey(0, kp0)] = landmarkIdx;
        track[makeTrackKey(1, kp1)] = landmarkIdx;

        observations.push_back({0, landmarkIdx, cv::Point2f(poseInlierPoints1[pointIdxInPoseInliers].x, poseInlierPoints1[pointIdxInPoseInliers].y)});
        observations.push_back({1, landmarkIdx, cv::Point2f(poseInlierPoints2[pointIdxInPoseInliers].x, poseInlierPoints2[pointIdxInPoseInliers].y)});

        pointIndexInPoints3D++;
    }

    imagesProcessed = 2;


    // =================== 链式扩展至 N 张图 =====================
    int lastRegisteredIdx = 1;

    for (int currentImageIdx = 2; currentImageIdx < pictureNum; currentImageIdx++)
    {
        int prevImageIdx = lastRegisteredIdx;

        std::vector<cuda_MatchResult> PrevCurrentMatches = cuda_FeatureMatch(features[prevImageIdx], features[currentImageIdx]);

        int matchPointsCount = countValidFeature(PrevCurrentMatches);
        std::cout << "\nimg" << prevImageIdx << "-img" << currentImageIdx << " matched points: " << matchPointsCount << std::endl;

        std::vector<cv::Point3f> pnpObjectPoints;   // PnP 使用的已有 3D 点
        std::vector<cv::Point2f> pnpImagePoints;    // 这些3D点在当前图像中的2D位置
        std::vector<NewMatch> newMatches;
        std::vector<PnPCorrespondence> pnpCorrespondences;

        assert(PrevCurrentMatches.size() == features[prevImageIdx].numFeatures);

        for (int PrevImagePointIdx = 0; PrevImagePointIdx < features[prevImageIdx].numFeatures; PrevImagePointIdx++)
        {
            int CurImagePointIdx = PrevCurrentMatches[PrevImagePointIdx].bestIdx;

            if (CurImagePointIdx < 0) continue;

            // 是匹配点，查 track
            auto it = track.find(makeTrackKey(prevImageIdx, PrevImagePointIdx));
            if (it != track.end())
            {
                int landmarkIdx = it->second;   // landmarkIdx：该特征点在 pointcloud 里面对应的索引
                // 取出 3D 点，作为 PnP 的 3D 输入
                pnpObjectPoints.push_back(pointCloud[landmarkIdx].position); 
                // 当前图像中与该 3D 点对应的 2D 特征点
                pnpImagePoints.emplace_back(features[currentImageIdx].kpX[CurImagePointIdx], features[currentImageIdx].kpY[CurImagePointIdx]);
                // 保存 3D 点与当前图像特征点之间的对应关系
                pnpCorrespondences.push_back({landmarkIdx, CurImagePointIdx});
            }
            else
            {
                // track 中未找到匹配的 3D 点，加入 newMatch
                newMatches.push_back({PrevImagePointIdx, CurImagePointIdx});
            }
        }
        
        std::cout << "Available 3D-2D points: " << pnpObjectPoints.size()
        << ", new points: " << newMatches.size() << std::endl;

        PnPResult pnpResult = solvePnPForNewView(pnpObjectPoints, pnpImagePoints, intr);
        // 若 PnP 没成功
        if (!pnpResult.success) 
        {
            std::cerr << "[SFM] img" << currentImageIdx << " PnP failed. Skip it." << std::endl;
            continue;
        }

        cameras[currentImageIdx].R = pnpResult.pose.R;
        cameras[currentImageIdx].t = pnpResult.pose.t;

        imagesProcessed++;
        lastRegisteredIdx = currentImageIdx;

        // 将新的匹配关系加入 track
        // inlierIndices：属于输入的 objectPoints / imagePoints 中的原始下标，只包含其中的有效点的下标
        for (int idx : pnpResult.inlierIndices)
        {
            int landmarkIdx = pnpCorrespondences[idx].landmarkIdx;
            int CurImagePointIdx = pnpCorrespondences[idx].CurImagePointIdx;

            track[makeTrackKey(currentImageIdx, CurImagePointIdx)] = landmarkIdx;

            cv::Point2f uv(features[currentImageIdx].kpX[CurImagePointIdx], features[currentImageIdx].kpY[CurImagePointIdx]);
            observations.push_back({currentImageIdx, landmarkIdx, cv::Point2f(uv.x, uv.y)});
        }

        // 准备新匹配的2D点
        std::vector<cv::Point2f> newPointsPrev;
        std::vector<cv::Point2f> newPointsCur;

        convertNewMatches(features[prevImageIdx], features[currentImageIdx], newMatches, newPointsPrev, newPointsCur);

        // 将新的匹配点进行三角化
        std::vector<int> Mask3D;
        std::vector<cv::Point3f> Points3D = cudaTriangulation(newPointsPrev, newPointsCur, intr, cameras[prevImageIdx], cameras[currentImageIdx], Mask3D);
        
        #if DEBUG_OUTPUT
            std::cout << "Mask3D: " << Mask3D.size() << std::endl;
            std::cout << "Points3D: " << Points3D.size() << std::endl;
            std::cout << "newPointsPrev: " << newPointsPrev.size() << std::endl;
            std::cout << "newPointsCur: " << newPointsCur.size() << std::endl;
        #endif


        int pointIndexInPoints3D = 0;
        for (int pointIdxInNewPoints = 0; pointIdxInNewPoints < Mask3D.size(); pointIdxInNewPoints++)
        {
            if (Mask3D[pointIdxInNewPoints] == 0) continue;

            int PrevImagePointIdx = newMatches[pointIdxInNewPoints].PrevImagePointIdx;
            int CurImagePointIdx  = newMatches[pointIdxInNewPoints].CurImagePointIdx;

            ColoredPoint3D point = getColorPoint3D(pointIndexInPoints3D, pointIdxInNewPoints, images[prevImageIdx], newPointsPrev, Points3D);
            int landmarkIdx = static_cast<int>(pointCloud.size());
            pointCloud.push_back(point);

            track[makeTrackKey(prevImageIdx, PrevImagePointIdx)] = landmarkIdx;
            track[makeTrackKey(currentImageIdx, CurImagePointIdx)] = landmarkIdx;

            observations.push_back({prevImageIdx, landmarkIdx, cv::Point2f(newPointsPrev[pointIdxInNewPoints].x, newPointsPrev[pointIdxInNewPoints].y)});
            observations.push_back({currentImageIdx, landmarkIdx, cv::Point2f(newPointsCur[pointIdxInNewPoints].x, newPointsCur[pointIdxInNewPoints].y)});

            pointIndexInPoints3D++;
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

