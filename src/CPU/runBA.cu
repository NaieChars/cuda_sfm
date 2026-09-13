#include "runBA.cuh"

#include "DataStruct.h"
#include "SFM.h"
#include "CameraIntrinsics.h"

#include "../GPU/d_BA.cuh"
#include "BundleAdjustment.h"

#include "Config.h"

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>


double computeReprojectionError(
    const SFMResult& result,
    const CameraIntrinsics& intr)
{
    double totalError = 0.0;
    int validCount = 0;

    for (const auto& obs : result.observations)
    {
        const CameraPose& cam = result.cameras[obs.cameraIdx];
        const cv::Point3f& P = result.pointCloud[obs.pointIdx].position;

        // 世界坐标 -> 相机坐标
        cv::Mat X = (cv::Mat_<double>(3, 1)
            << P.x, P.y, P.z);

        cv::Mat Xc = cam.R * X + cam.t;

        double Xc_x = Xc.at<double>(0);
        double Xc_y = Xc.at<double>(1);
        double Xc_z = Xc.at<double>(2);

        // 点在相机后面，跳过
        if (Xc_z <= 0.0)
            continue;

        // 相机坐标 -> 像素坐标
        double u = intr.fx * Xc_x / Xc_z + intr.cx;
        double v = intr.fy * Xc_y / Xc_z + intr.cy;

        double dx = u - obs.uv.x;
        double dy = v - obs.uv.y;

        double error = std::sqrt(dx * dx + dy * dy);

        totalError += error;
        validCount++;
    }

    if (validCount == 0)
        return 0.0;

    return totalError / validCount;
}



void runBA(
    SFMResult& result,
    const CameraIntrinsics& intr
)
{
    double errorBefore = computeReprojectionError(result, intr);

    std::cout << "\n========== BA ==========\n";
    std::cout << "Reprojection Error Before: "
            << errorBefore << " px\n";


    int numCameras =
        static_cast<int>(result.cameras.size());

    int numPoints =
        static_cast<int>(result.pointCloud.size());

    int numObservations =
        static_cast<int>(result.observations.size());

    std::cout << "\n========== BA ==========\n";
    std::cout << "Cameras: " << numCameras << std::endl;
    std::cout << "Points: " << numPoints << std::endl;
    std::cout << "Observations: " << numObservations << std::endl;


    // ========================================================
    // 1. CPU 数据转换
    // ========================================================

    std::vector<d_CameraPose> camerasGPU(numCameras);
    std::vector<d_Observation> observationsGPU(numObservations);
    std::vector<cv::Point3f> pointsGPU(numPoints);

    for (int i = 0; i < numCameras; ++i)
    {
        const CameraPose& cam = result.cameras[i];

        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                camerasGPU[i].t[r] = static_cast<float>(cam.t.at<double>(r, 0));
            }
        }

        for (int r = 0; r < 3; ++r)
        {
            camerasGPU[i].t[r] =
                static_cast<float>(cam.t.at<double>(r, 0));
        }
    }

    for (int i = 0; i < numPoints; ++i)
    {
        pointsGPU[i] = cv::Point3f(
            result.pointCloud[i].position.x,
            result.pointCloud[i].position.y,
            result.pointCloud[i].position.z
        );
    }

    for(int i=0;i<numObservations;++i)
    {
        const Observation& obs = result.observations[i];

        observationsGPU[i].cameraId = obs.cameraIdx;
        observationsGPU[i].pointId = obs.pointIdx;

        // Pixel coordinates -> normalized camera coordinates
        observationsGPU[i].u =
            (obs.uv.x - intr.cx) / intr.fx;

        observationsGPU[i].v =
            (obs.uv.y - intr.cy) / intr.fy;
    }


    // ========================================================
    // 2. 建立 Observation 索引
    // ========================================================

    std::vector<int> pointObsStart;
    std::vector<int> pointObsCount;
    std::vector<int> pointObsIndices;

    std::vector<int> cameraObsStart;
    std::vector<int> cameraObsCount;
    std::vector<int> cameraObsIndices;

    buildPointObservationIndex(
        observationsGPU,
        numPoints,
        pointObsStart,
        pointObsCount,
        pointObsIndices
    );

    buildCameraObservationIndex(
        observationsGPU,
        numCameras,
        cameraObsStart,
        cameraObsCount,
        cameraObsIndices
    );


    // ========================================================
    // 3. 上传 GPU
    // ========================================================

    BAGPUData gpu;

    uploadBADataToGPU(
        observationsGPU,
        camerasGPU,
        pointsGPU,

        pointObsStart,
        pointObsCount,
        pointObsIndices,

        cameraObsStart,
        cameraObsCount,
        cameraObsIndices,

        numObservations,
        numCameras,
        numPoints,

        gpu
    );


    // ========================================================
    // 4. Compute Observation Jacobian
    // ========================================================

    int blockSize = 256;

    int gridObs =
        (numObservations + blockSize - 1) /
        blockSize;

    computeObsJacobianKernel<<<gridObs, blockSize>>>(
        gpu.d_observations,
        gpu.d_cameras,
        reinterpret_cast<const float3*>(gpu.d_points),
        numObservations,
        reinterpret_cast<float2*>(gpu.d_residuals),
        gpu.d_Jp,
        gpu.d_Jc
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int gridPoints = (numPoints + blockSize - 1) / blockSize;

    computePointBlocksKernel<<<gridPoints, blockSize>>>(
        gpu.d_observations,
        reinterpret_cast<const float2*>(gpu.d_residuals),
        gpu.d_Jp,
        gpu.d_pointObsStart,
        gpu.d_pointObsCount,
        gpu.d_pointObsIndices,
        numPoints,
        gpu.d_Hpp,
        gpu.d_gp
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int gridCameras = (numCameras + blockSize - 1) / blockSize;

    computeCameraBlocksKernel<<<gridCameras, blockSize>>>(
        gpu.d_observations,
        reinterpret_cast<const float2*>(gpu.d_residuals),
        gpu.d_Jp,
        gpu.d_Jc,
        gpu.d_cameraObsStart,
        gpu.d_cameraObsCount,
        gpu.d_cameraObsIndices,
        numCameras,
        gpu.d_Hcc,
        gpu.d_gc,
        gpu.d_M
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    invertPointBlocksKernel<<<gridPoints, blockSize>>>(
        gpu.d_Hpp,
        gpu.d_HppInv,
        numPoints
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    int cameraDim = numCameras * 6;

    float* d_schurH = nullptr;
    float* d_schurB = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_schurH,
        cameraDim * cameraDim * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_schurB,
        cameraDim * sizeof(float)
    ));

    CUDA_CHECK(cudaMemset(
        d_schurH,
        0,
        cameraDim * cameraDim * sizeof(float)
    ));

    CUDA_CHECK(cudaMemset(
        d_schurB,
        0,
        cameraDim * sizeof(float)
    ));


    addCameraBlocksToSchurKernel<<<gridCameras, blockSize>>>(
        gpu.d_Hcc,
        gpu.d_gc,
        numCameras,
        d_schurH,
        d_schurB
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    int gridSchur = (numPoints + blockSize - 1) / blockSize;

    computeSchurContributionKernel<<<gridSchur, blockSize>>>(
        gpu.d_observations,
        gpu.d_M,
        gpu.d_HppInv,
        gpu.d_gp,

        gpu.d_pointObsStart,
        gpu.d_pointObsCount,
        gpu.d_pointObsIndices,

        numPoints,
        numCameras,

        d_schurH,
        d_schurB
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> schurH(
    cameraDim * cameraDim);

    std::vector<float> schurB(
        cameraDim
    );

    CUDA_CHECK(cudaMemcpy(
        schurH.data(),
        d_schurH,
        cameraDim * cameraDim * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        schurB.data(),
        d_schurB,
        cameraDim * sizeof(float),
        cudaMemcpyDeviceToHost
    ));


    // ---------------- 调试输出 -----------------
    std::cout << "\nSchur H first 6x6:\n";

    for (int i = 0; i < std::min(6, cameraDim); ++i)
    {
        for (int j = 0; j < std::min(6, cameraDim); ++j)
        {
            std::cout << schurH[i * cameraDim + j]
                    << " ";
        }

        std::cout << std::endl;
    }

    std::cout << "\nSchur B first 6:\n";

    for (int i = 0; i < std::min(6, cameraDim); ++i)
    {
        std::cout << schurB[i] << " ";
    }

    std::cout << std::endl;

    // ============================================================
    // Solve Schur system
    // Camera 0 is fixed
    // ============================================================

    int fixedCameraDim = 6;
    int solveDim = cameraDim - fixedCameraDim;

    Eigen::MatrixXf S(solveDim, solveDim);
    Eigen::VectorXf b(solveDim);

    // Remove Camera 0's 6 parameters
    for (int r = 0; r < solveDim; ++r)
    {
        int srcR = r + fixedCameraDim;

        b(r) = schurB[srcR];

        for (int c = 0; c < solveDim; ++c)
        {
            int srcC = c + fixedCameraDim;

            S(r, c) =
                schurH[srcR * cameraDim + srcC];
        }
    }

    // Solve:
    // S * deltaCamera = b

    Eigen::VectorXf deltaCamera =
        S.ldlt().solve(-b);

    std::cout << "\nDelta Camera first 12:\n";

    for (int i = 0; i < std::min(12, solveDim); ++i)
    {
        std::cout << deltaCamera(i) << " ";
    }

    std::cout << std::endl;

    // ============================================================
    // Update camera poses
    // Camera 0 is fixed
    // ============================================================

    for (int camIdx = 1; camIdx < numCameras; ++camIdx)
    {
        int offset = (camIdx - 1) * 6;

        // Rotation increment
        cv::Mat deltaRvec = (cv::Mat_<double>(3, 1)
            << deltaCamera(offset + 0),
            deltaCamera(offset + 1),
            deltaCamera(offset + 2));

        cv::Mat deltaR;
        cv::Rodrigues(deltaRvec, deltaR);

        // Translation increment
        cv::Mat deltaT = (cv::Mat_<double>(3, 1)
            << deltaCamera(offset + 3),
            deltaCamera(offset + 4),
            deltaCamera(offset + 5));

        // Current pose
        cv::Mat R = result.cameras[camIdx].R;
        cv::Mat t = result.cameras[camIdx].t;

        // Left-multiplicative SE(3) update
        result.cameras[camIdx].R = deltaR * R;
        result.cameras[camIdx].t = deltaR * t + deltaT;
    }

    #if DEBUG_OUTPUT
        std::cout << "\nUpdated Camera 1:\n";
        std::cout << "R =\n" << result.cameras[1].R << std::endl;
        std::cout << "t =\n" << result.cameras[1].t << std::endl;
    #endif



    float* d_deltaCamera = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_deltaCamera,
        cameraDim * sizeof(float)
    ));

    std::vector<float> deltaCameraGPU(cameraDim, 0.0f);

    for (int camIdx = 1; camIdx < numCameras; ++camIdx)
    {
        int srcOffset = (camIdx - 1) * 6;
        int dstOffset = camIdx * 6;

        for (int k = 0; k < 6; ++k)
        {
            deltaCameraGPU[dstOffset + k] =
                deltaCamera(srcOffset + k);
        }
    }

    CUDA_CHECK(cudaMemcpy(
        d_deltaCamera,
        deltaCameraGPU.data(),
        cameraDim * sizeof(float),
        cudaMemcpyHostToDevice
    ));


    computePointUpdateKernel<<<gridPoints, blockSize>>>(
        gpu.d_observations,
        gpu.d_M,
        gpu.d_HppInv,
        gpu.d_gp,

        gpu.d_pointObsStart,
        gpu.d_pointObsCount,
        gpu.d_pointObsIndices,

        d_deltaCamera,

        numPoints,

        reinterpret_cast<float3*>(gpu.d_points),
        1.0f
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<cv::Point3f> updatedPoints(numPoints);

    CUDA_CHECK(cudaMemcpy(
        updatedPoints.data(),
        gpu.d_points,
        numPoints * sizeof(cv::Point3f),
        cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < numPoints; ++i)
    {
        result.pointCloud[i].position =
            updatedPoints[i];
    }

    CUDA_CHECK(cudaFree(d_deltaCamera));

    double errorAfter = computeReprojectionError(result, intr);

    std::cout << "Reprojection Error After : "
            << errorAfter << " px\n";

    std::cout << "Error Change             : "
            << errorAfter - errorBefore << " px\n";


    std::cout << "Jacobian computation finished."
              << std::endl;
}