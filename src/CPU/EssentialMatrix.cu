#include "EssentialMatrix.cuh"

// ----------------------- 外部接口 --------------------------

RansacResult estimateEssentialMatrix(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
        const CameraIntrinsics& intr)
{
    RansacResult result;

    result.E = cv::findEssentialMat(
        points1, points2,
        intr.K,
        cv::RANSAC,
        0.999,
        1.5,
        result.essentialMask
    );

    std::cout << "[EssentialMat] Inliers: " << cv::countNonZero(result.essentialMask) << " / " << points1.size() << std::endl;

    return result;
}


RansacResult estimateEssentialMatrixRANSAC(const std::vector<cv::Point2f>& points1,
                        const std::vector<cv::Point2f>& points2,
                        const CameraIntrinsics& intr, 
                        const float confidence,
                        const int maxIterations)
{
    RansacResult result;

    const int n = static_cast<int>(points1.size());
    assert(points1.size() == points2.size());
    assert(n >= 8);

    result.essentialMask.resize(n);

    // 点归一化
    std::vector<cv::Point2f> normPoints1 = normalizePoints(points1, intr.fx, intr.fy, intr.cx, intr.cy);
    std::vector<cv::Point2f> normPoints2 = normalizePoints(points2, intr.fx, intr.fy, intr.cx, intr.cy);


    // CUDA
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    float* d_points1 = nullptr;
    float* d_points2 = nullptr;
    double* E = nullptr;
    unsigned char* d_inlierMask = nullptr;
    int* d_inlierCount = nullptr;

    const size_t bytes1 = n * sizeof(cv::Point2f);
    CUDA_CHECK(cudaMalloc(&d_points1, bytes1));
    CUDA_CHECK(cudaMalloc(&d_points2, bytes1));
    CUDA_CHECK(cudaMalloc(&E, 9 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_inlierMask, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_inlierCount, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_inlierCount, 0, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points1, normPoints1.data(), n * sizeof(cv::Point2f), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points2, normPoints2.data(), n * sizeof(cv::Point2f), cudaMemcpyHostToDevice));

    // RANSAC 主循环
    const int sampleSize = 8;
    int bestInlierCount = 0;
    int actualIterations = maxIterations; // 当前允许的最大迭代次数，可能根据内点比例动态缩短
    cv::Mat bestE;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int iter = 0; iter < maxIterations; iter++)
    {
        // CPU 随机选取8点
        std::vector<cv::Point2f> sample1, sample2;
        sample1.reserve(sampleSize);
        sample2.reserve(sampleSize);
        sample8Points(normPoints1, normPoints2, sample1, sample2, gen);

        // 8点法求候选E
        cv::Mat candidateE = generateCandidateE(sample1, sample2);
        std::array<double, 9> E_flat = flattenMatrix_3x3(candidateE);

        // GPU 验证
        CUDA_CHECK(cudaMemcpy(E, E_flat.data(), 9 * sizeof(double), cudaMemcpyHostToDevice));
        

        CUDA_CHECK(cudaMemset(d_inlierCount, 0, sizeof(int)));

        validateEssentialMatrixCUDA<<<gridSize, blockSize>>>(E, d_points1, d_points2, d_inlierMask, d_inlierCount, n, 1e-6);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 先只拿回当前E的匹配内点数
        int currentInlierCount;
        CUDA_CHECK(cudaMemcpy(&currentInlierCount, d_inlierCount, sizeof(int), cudaMemcpyDeviceToHost));
        
        // 内点数更多，更新
        if (currentInlierCount > bestInlierCount)
        {
            bestE = candidateE;
            bestInlierCount = currentInlierCount;


            // 拿回掩码
            CUDA_CHECK(cudaMemcpy(result.essentialMask.data(), d_inlierMask, n * sizeof(unsigned char), cudaMemcpyDeviceToHost));

            // 动态计算，提前终止
            float w = static_cast<float>(bestInlierCount) / static_cast<float>(n);
            if (w > 0.0f && w < 1.0f)
            {
                double wd = static_cast<double>(w);
                double denom = std::log(1.0 - std::pow(wd, sampleSize));
                if (denom < -1e-12)
                {
                    double requiredIters = std::log(1.0 - static_cast<double>(confidence)) / denom;
                    if (requiredIters < static_cast<double>(actualIterations))
                    {
                        int requiredItersInt = static_cast<int>(std::ceil(requiredIters));
                        actualIterations = std::max(requiredItersInt, iter + 1);
                    }
                }
            }
        }
        if (iter + 1 >= actualIterations)
        {
            actualIterations = iter + 1;
            break;
        }
    }

    result.iterationsUsed = actualIterations;

    std::cout << "[CUDA RANSAC] Inliers: "
          << bestInlierCount << " / " << points1.size() << std::endl;

    if (bestInlierCount < sampleSize)
    {
        std::cerr << "[RANSAC_CUDA] Warning: Best inlier count too small (<8), essential matrix estimation is unreliable" << std::endl;
        return result;
    }

    // 用全部内点重新再拟合一次E，再做一次kernel，传回最终essentialMask
    std::vector<cv::Point2f> inlierPts1, inlierPts2;
    inlierPts1.reserve(bestInlierCount);
    inlierPts2.reserve(bestInlierCount);
    inlierPts1 = fliterInlierPoints(normPoints1, result.essentialMask);
    inlierPts2 = fliterInlierPoints(normPoints2, result.essentialMask);
    
    bestE = refineEssentialMatrix(inlierPts1, inlierPts2);
    CUDA_CHECK(cudaMemcpy(E, flattenMatrix_3x3(bestE).data(), 9 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_inlierCount, 0, sizeof(int)));

    validateEssentialMatrixCUDA<<<gridSize, blockSize>>>(E, d_points1, d_points2, d_inlierMask, d_inlierCount, n, 1e-6); 
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize()); 
    CUDA_CHECK(cudaMemcpy(result.essentialMask.data(), d_inlierMask, n * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    result.E = bestE;

    std::cout << "[EssentialMat] Inliers: " << cv::countNonZero(result.essentialMask) << " / " << points1.size() << std::endl;

    return result;
}


// 8组归一化匹配点，输出一个候选E
cv::Mat generateCandidateE(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2)
{
    const int n = static_cast<int>(points1.size());
    assert(points1.size() == points2.size());
    assert(n >= 8);

    // 构造A
    cv::Mat A(n ,9, CV_64F);
    for (int i = 0; i < n; i++)
    {
        double x  = points1[i].x;
        double y  = points1[i].y;
        double xp = points2[i].x;
        double yp = points2[i].y;

        A.at<double>(i, 0) = xp * x;
        A.at<double>(i, 1) = xp * y;
        A.at<double>(i, 2) = xp;

        A.at<double>(i, 3) = yp * x;
        A.at<double>(i, 4) = yp * y;
        A.at<double>(i, 5) = yp;

        A.at<double>(i, 6) = x;
        A.at<double>(i, 7) = y;
        A.at<double>(i, 8) = 1.0;
    }

    cv::SVD svdA(A, cv::SVD::FULL_UV);
    cv::Mat e = svdA.vt.row(8).clone();

    cv::Mat E = e.reshape(1, 3);
    cv::SVD svdE(E, cv::SVD::FULL_UV);

    double sigma =(svdE.w.at<double>(0) + svdE.w.at<double>(1)) / 2.0;

    cv::Mat Sigma = cv::Mat::zeros(3, 3, CV_64F);
    Sigma.at<double>(0, 0) = sigma;
    Sigma.at<double>(1, 1) = sigma;

    cv::Mat Erefined = svdE.u * Sigma * svdE.vt;

    return Erefined;
}


// 输入多个已经归一化的匹配点，重新拟合最终E
// 该函数内部用的 A^T A 方案，数值稳定性比对A进行SVD较差
cv::Mat refineEssentialMatrix(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2)
{
    const int n = static_cast<int>(points1.size());

    assert(points1.size() == points2.size());
    assert(n >= 8);

    // 构造 A (n × 9)
    cv::Mat A(n, 9, CV_64F);

    for (int i = 0; i < n; i++)
    {
        double x  = points1[i].x;
        double y  = points1[i].y;
        double xp = points2[i].x;
        double yp = points2[i].y;

        A.at<double>(i, 0) = xp * x;
        A.at<double>(i, 1) = xp * y;
        A.at<double>(i, 2) = xp;

        A.at<double>(i, 3) = yp * x;
        A.at<double>(i, 4) = yp * y;
        A.at<double>(i, 5) = yp;

        A.at<double>(i, 6) = x;
        A.at<double>(i, 7) = y;
        A.at<double>(i, 8) = 1.0;
    }

    // A^T A，只剩下 9×9
    cv::Mat AtA = A.t() * A;

    // 对 9×9 矩阵做 SVD
    cv::SVD svd(AtA, cv::SVD::FULL_UV);

    // 最小特征值对应的特征向量
    cv::Mat e = svd.vt.row(8).clone();

    // 1×9 → 3×3
    cv::Mat E = e.reshape(1, 3).clone();

    // 强制 E 为 rank 2
    cv::SVD svdE(E, cv::SVD::FULL_UV);

    double sigma =
        (svdE.w.at<double>(0) + svdE.w.at<double>(1)) / 2.0;

    cv::Mat Sigma = cv::Mat::zeros(3, 3, CV_64F);
    Sigma.at<double>(0, 0) = sigma;
    Sigma.at<double>(1, 1) = sigma;

    cv::Mat Erefined = svdE.u * Sigma * svdE.vt;

    return Erefined;
}


// 输入已经过归一化的点坐标系列
// (该函数有一个优化，将随机数生成器提取到外面)
void sample8Points(const std::vector<cv::Point2f>& points1, 
                   const std::vector<cv::Point2f>& points2,
                   std::vector<cv::Point2f>& sample1,
                   std::vector<cv::Point2f>& sample2,
                   std::mt19937& gen)
{
    // 随机索引范围 [0, n1 - 1]
    std::uniform_int_distribution<int> dist(0, static_cast<int>(points1.size()) - 1);

    // 用 std::set 处理重复抽取
    std::set<int> selectedIndices;
    while (selectedIndices.size() < 8)
    {
        int index = dist(gen);  // 随机抽数
        selectedIndices.insert(index);
    }


    for (int index : selectedIndices)
    {
        sample1.push_back(points1[index]);
        sample2.push_back(points2[index]);
    }
}
