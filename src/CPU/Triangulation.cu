#include "Triangulation.cuh"
#include "cuda_Check.cuh"
#include "../GPU/TriangulationCUDA.cuh"
#include "util.h"
#include <cuda_runtime.h>

//第二次掩码筛选复用 PoseRecovery.h 里的 fliter 函数


std::vector<cv::Point3f> cudaTriangulation(
    const std::vector<cv::Point2f>& inlierPoints1,
    const std::vector<cv::Point2f>& inlierPoints2,
    const CameraIntrinsics& intr,
    const CameraPose& pose1,
    const CameraPose& pose2,
    std::vector<int>& Mask3D) 
{
    // P1, P2
    cv::Mat P1(3, 4, CV_64F);
    P1.setTo(cv::Scalar(0));

    pose1.R.copyTo(P1(cv::Rect(0, 0, 3, 3)));
    pose1.t.copyTo(P1(cv::Rect(3, 0, 1, 3)));

    P1 = intr.K * P1;

    cv::Mat P2(3, 4, CV_64F);
    P2.setTo(cv::Scalar(0));

    pose2.R.copyTo(P2(cv::Rect(0, 0, 3, 3)));
    pose2.t.copyTo(P2(cv::Rect(3, 0, 1, 3)));

    P2 = intr.K * P2;

    // ------------ 调试输出: 检查P1,P2 ------------
    #if DEBUG_OUTPUT
        std::cout << "P1:\n" << P1 << "\n";
        std::cout << "P2:\n" << P2 << "\n";
    #endif

    // ---------------调试：CPU 端 Opencv 三角化对照 ----------------
    
    #if DEBUG_OUTPUT
        cv::Mat points4D;
        cv::triangulatePoints(P1, P2, inlierPoints1, inlierPoints2, points4D);
        points4D.convertTo(points4D, CV_64F);
        for (int i = 0; i < std::min(10, points4D.cols); ++i)
        {
            double w = points4D.at<double>(3, i);
            double x = points4D.at<double>(0, i) / w;
            double y = points4D.at<double>(1, i) / w;
            double z = points4D.at<double>(2, i) / w;

            double z2 = P2.at<double>(2, 0) * x +
                    P2.at<double>(2, 1) * y +
                    P2.at<double>(2, 2) * z +
                    P2.at<double>(2, 3);

            std::cout << "i=" << i
                    << " xyz=(" << x << "," << y << "," << z << ")"
                    << " z1=" << z << " z2=" << z2 << '\n';
        }
    #endif
    // ------------------------------------------------------------------------


    // P1, P2 -> flatten
    std::array<double, 12> P1_flat;
    P1_flat = flattenMatrix_3x4(P1);
    std::array<double, 12> P2_flat;
    P2_flat = flattenMatrix_3x4(P2);

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

    // ------- 调试序列：检查三角化每一步失败点数 ---------
    int* d_failSolve;
    int* d_failW;
    int* d_failDepth1;
    int* d_failDepth2;
    int* d_failReproj;

    CUDA_CHECK(cudaMalloc(&d_failSolve, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_failW, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_failDepth1, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_failDepth2, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_failReproj, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_failSolve, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_failW, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_failDepth1, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_failDepth2, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_failReproj, 0, sizeof(int)));
    // ------------------------------------------------------

    CUDA_CHECK(cudaMalloc(&d_points3D, numPoints * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Mask3D, numPoints * sizeof(int)));

    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    
    triangulateKernel<<<gridSize, blockSize>>>(d_points1, d_points2, d_P1, d_P2, numPoints, d_points3D, 1.5, d_Mask3D, d_failSolve, d_failW, d_failDepth1, d_failDepth2, d_failReproj);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU -> CPU
    std::vector<cv::Point3f> h_points3D(numPoints);
    std::vector<int> h_Mask3D(numPoints);

    CUDA_CHECK(cudaMemcpy(h_points3D.data(), d_points3D, static_cast<size_t>(numPoints * 3 * sizeof(float)), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Mask3D.data(), d_Mask3D, static_cast<size_t>(numPoints * sizeof(int)), cudaMemcpyDeviceToHost));

    // ----------------- 将调试序列拷回CPU并输出 -----------------
    int h_failSolve;
    int h_failW;
    int h_failDepth1;
    int h_failDepth2;
    int h_failReproj;

    CUDA_CHECK(cudaMemcpy(&h_failSolve, d_failSolve, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_failW, d_failW, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_failDepth1, d_failDepth1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_failDepth2, d_failDepth2, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_failReproj, d_failReproj, sizeof(int), cudaMemcpyDeviceToHost));

    #if DEBUG_OUTPUT
        std::cout << "failSolve  = " << h_failSolve << std::endl;
        std::cout << "failW      = " << h_failW << std::endl;
        std::cout << "failDepth1 = " << h_failDepth1 << std::endl;
        std::cout << "failDepth2 = " << h_failDepth2 << std::endl;
        std::cout << "failReproj = " << h_failReproj << std::endl;
    #endif
    // -----------------------------------------------------------------------------

    // 过滤失败点（经深度验证与重投影误差验证）
    std::vector<cv::Point3f> finalPoints_3d;
    for (int i = 0; i < h_Mask3D.size(); i++)
    {
        if (h_Mask3D[i] == 0)
            continue;
        
        finalPoints_3d.push_back(h_points3D[i]);
    }
    Mask3D = h_Mask3D;

    int finalPointsNum = finalPoints_3d.size();
    std::cout << "[Triangulation] Valid 3D points: " << finalPointsNum << std::endl;

    CUDA_CHECK(cudaFree(d_P1));
    CUDA_CHECK(cudaFree(d_P2));
    CUDA_CHECK(cudaFree(d_points1));
    CUDA_CHECK(cudaFree(d_points2));
    CUDA_CHECK(cudaFree(d_points3D));
    CUDA_CHECK(cudaFree(d_Mask3D));
    
    // ------------------ 释放调试序列 --------------
    cudaFree(d_failSolve);
    cudaFree(d_failW);
    cudaFree(d_failDepth1);
    cudaFree(d_failDepth2);
    cudaFree(d_failReproj);

    return finalPoints_3d;
}