#include "Triangulation.cuh"
#include "cuda_Check.cuh"
#include "../GPU/TriangulationCUDA.cuh"
#include <cuda_runtime.h>

//第二次掩码筛选复用 PoseRecovery.h 里的 fliter 函数

// flatten Matrix(3 x 4)
std::array<double, 12> flattenMatrix(const cv::Mat& P)
{
    std::array<double, 12> flat;

    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 4; c++)
            flat[r * 4 + c] = P.at<double>(r, c);
    }
    return flat;
}


// cuda_triangulation 核心流程
// InlierPoints1是已经二次筛选的
std::vector<cv::Point3f> cudaTriangulation(
    const std::vector<cv::Point2f>& inlierPoints1,
    const std::vector<cv::Point2f>& inlierPoints2,
    const CameraIntrinsics& intr,
    const cv::Mat& R, const cv::Mat& t,
    std::vector<int>& Mask3D)
{
    // P1, P2
    cv::Mat P1(3, 4, CV_64F);
    P1.setTo(cv::Scalar(0));
    P1.at<double>(0, 0) = 1.0;
    P1.at<double>(1, 1) = 1.0;
    P1.at<double>(2, 2) = 1.0;
    P1 = intr.K * P1; 

    cv::Mat P2(3, 4, CV_64F);
    R.copyTo(P2(cv::Rect(0, 0, 3, 3)));   
    t.copyTo(P2(cv::Rect(3, 0, 1, 3)));  
    P2 = intr.K * P2; 

    // P1, P2 -> flatten
    std::array<double, 12> P1_flat;
    P1_flat = flattenMatrix(P1);
    std::array<double, 12> P2_flat;
    P2_flat = flattenMatrix(P2);

    // CPU -> GPU
    double* d_P1 = nullptr;
    double* d_P2 = nullptr;
    size_t P_Bytes = 12 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_P1, P_Bytes));
    CUDA_CHECK(cudaMalloc(&d_P2, P_Bytes));
    CUDA_CHECK(cudaMemcpy(d_P1, P1_flat.data(), P_Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P2, P2_flat.data(), P_Bytes, cudaMemcpyHostToDevice));
    float* d_points1 = nullptr;
    float* d_points2 = nullptr;
    size_t point1_Bytes = inlierPoints1.size() * sizeof(cv::Point2f);
    size_t point2_Bytes = inlierPoints2.size() * sizeof(cv::Point2f);
    CUDA_CHECK(cudaMalloc(&d_points1, point1_Bytes));
    CUDA_CHECK(cudaMalloc(&d_points2, point2_Bytes));
    CUDA_CHECK(cudaMemcpy(d_points1, inlierPoints1.data(), point1_Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points2, inlierPoints2.data(), point2_Bytes, cudaMemcpyHostToDevice));

    // kernel
    int numPoints = inlierPoints1.size();
    float* d_points3D = nullptr;
    int* d_Mask3D = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points3D, numPoints * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Mask3D, numPoints * sizeof(int)));

    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    
    triangulateKernel<<<gridSize, blockSize>>>(d_points1, d_points2, d_P1, d_P2, numPoints, d_points3D, 1.5, d_Mask3D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU -> CPU
    std::vector<cv::Point3f> h_points3D(numPoints);
    std::vector<int> h_Mask3D(numPoints);

    CUDA_CHECK(cudaMemcpy(h_points3D.data(), d_points3D, static_cast<size_t>(numPoints * 3 * sizeof(float)), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Mask3D.data(), d_Mask3D, static_cast<size_t>(numPoints * sizeof(int)), cudaMemcpyDeviceToHost));

    // 过滤失败点（经深度验证与重投影误差验证）
    std::vector<cv::Point3f> finalPoints_3d;
    for (int i = 0; i < h_Mask3D.size(); i++)
    {
        if (h_Mask3D[i] == 0)
            continue;
        
        finalPoints_3d.push_back(h_points3D[i]);
    }
    Mask3D = h_Mask3D;

    CUDA_CHECK(cudaFree(d_P1));
    CUDA_CHECK(cudaFree(d_P2));
    CUDA_CHECK(cudaFree(d_points1));
    CUDA_CHECK(cudaFree(d_points2));
    CUDA_CHECK(cudaFree(d_points3D));
    CUDA_CHECK(cudaFree(d_Mask3D));
    
    return finalPoints_3d;
}