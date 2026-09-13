#include "BundleAdjustment.h"

void buildPointObservationIndex(
    const std::vector<d_Observation>& observations,
    int numPoints,
    std::vector<int>& pointObsStart,
    std::vector<int>& pointObsCount,
    std::vector<int>& pointObsIndices)
{
    pointObsStart.resize(numPoints);
    pointObsCount.assign(numPoints, 0);

    // 第一次遍历：统计每个 Point 有多少 Observation
    for (int i = 0; i < observations.size(); ++i)
    {
        int pointId = observations[i].pointId;
        pointObsCount[pointId]++;
    }

    // 计算每个 Point 在 pointObsIndices 中的起始位置
    int offset = 0;

    for (int i = 0; i < numPoints; ++i)
    {
        pointObsStart[i] = offset;
        offset += pointObsCount[i];
    }

    pointObsIndices.resize(observations.size());

    // 临时记录每个 Point 当前已经填了几个
    std::vector<int> currentCount(numPoints, 0);

    // 第二次遍历：建立真正的索引
    for (int obsIdx = 0; obsIdx < observations.size(); ++obsIdx)
    {
        int pointId = observations[obsIdx].pointId;

        int index =
            pointObsStart[pointId] +
            currentCount[pointId];

        pointObsIndices[index] = obsIdx;

        currentCount[pointId]++;
    }
}


void buildCameraObservationIndex(
    const std::vector<d_Observation>& observations,
    int numCameras,
    std::vector<int>& cameraObsStart,
    std::vector<int>& cameraObsCount,
    std::vector<int>& cameraObsIndices)
{
    cameraObsStart.resize(numCameras);
    cameraObsCount.assign(numCameras, 0);

    // 统计每个 Camera 有多少 Observation
    for (int i = 0; i < observations.size(); ++i)
    {
        int cameraId = observations[i].cameraId;
        cameraObsCount[cameraId]++;
    }

    // 计算每个 Camera 的起始位置
    int offset = 0;

    for (int i = 0; i < numCameras; ++i)
    {
        cameraObsStart[i] = offset;
        offset += cameraObsCount[i];
    }

    cameraObsIndices.resize(observations.size());

    // 当前每个 Camera 已经填入多少 Observation
    std::vector<int> currentCount(numCameras, 0);

    // 建立索引
    for (int obsIdx = 0; obsIdx < observations.size(); ++obsIdx)
    {
        int cameraId = observations[obsIdx].cameraId;

        int index =
            cameraObsStart[cameraId] +
            currentCount[cameraId];

        cameraObsIndices[index] = obsIdx;

        currentCount[cameraId]++;
    }
}


void uploadBADataToGPU(
    const std::vector<d_Observation>& observations,
    const std::vector<d_CameraPose>& cameras,
    const std::vector<cv::Point3f>& points,
    const std::vector<int>& pointObsStart,
    const std::vector<int>& pointObsCount,
    const std::vector<int>& pointObsIndices,
    const std::vector<int>& cameraObsStart,
    const std::vector<int>& cameraObsCount,
    const std::vector<int>& cameraObsIndices,
    int numObservations,
    int numCameras,
    int numPoints,
    BAGPUData& gpu)
{
     CUDA_CHECK(cudaMalloc(
        &gpu.d_observations,
        numObservations * sizeof(d_Observation)));

    CUDA_CHECK(cudaMalloc(
        &gpu.d_cameras,
        numCameras * sizeof(d_CameraPose)));

    CUDA_CHECK(cudaMalloc(
        &gpu.d_points,
        numPoints * sizeof(cv::Point3f)));

    CUDA_CHECK(cudaMemcpy(
        gpu.d_observations,
        observations.data(),
        numObservations * sizeof(d_Observation),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        gpu.d_cameras,
        cameras.data(),
        numCameras * sizeof(d_CameraPose),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        gpu.d_points,
        points.data(),
        numPoints * sizeof(cv::Point3f),
        cudaMemcpyHostToDevice));


    // =========================================================
    // 2. Point -> Observation 索引
    // =========================================================

    CUDA_CHECK(cudaMalloc(
        &gpu.d_pointObsStart,
        numPoints * sizeof(int)));

    CUDA_CHECK(cudaMalloc(
        &gpu.d_pointObsCount,
        numPoints * sizeof(int)));

    CUDA_CHECK(cudaMalloc(
        &gpu.d_pointObsIndices,
        numObservations * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        gpu.d_pointObsStart,
        pointObsStart.data(),
        numPoints * sizeof(int),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        gpu.d_pointObsCount,
        pointObsCount.data(),
        numPoints * sizeof(int),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        gpu.d_pointObsIndices,
        pointObsIndices.data(),
        numObservations * sizeof(int),
        cudaMemcpyHostToDevice));


    // =========================================================
    // 3. Camera -> Observation 索引
    // =========================================================

    CUDA_CHECK(cudaMalloc(
        &gpu.d_cameraObsStart,
        numCameras * sizeof(int)));

    CUDA_CHECK(cudaMalloc(
        &gpu.d_cameraObsCount,
        numCameras * sizeof(int)));

    CUDA_CHECK(cudaMalloc(
        &gpu.d_cameraObsIndices,
        numObservations * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        gpu.d_cameraObsStart,
        cameraObsStart.data(),
        numCameras * sizeof(int),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        gpu.d_cameraObsCount,
        cameraObsCount.data(),
        numCameras * sizeof(int),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        gpu.d_cameraObsIndices,
        cameraObsIndices.data(),
        numObservations * sizeof(int),
        cudaMemcpyHostToDevice));


    // =========================================================
    // 4. Jacobian / Residual
    // =========================================================

    CUDA_CHECK(cudaMalloc(
        &gpu.d_residuals,
        numObservations * sizeof(float2)));

    CUDA_CHECK(cudaMalloc(
        &gpu.d_Jp,
        numObservations * 6 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &gpu.d_Jc,
        numObservations * 12 * sizeof(float)));


    // =========================================================
    // 5. Point blocks
    // =========================================================

    CUDA_CHECK(cudaMalloc(
        &gpu.d_Hpp,
        numPoints * 9 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &gpu.d_gp,
        numPoints * 3 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &gpu.d_HppInv,
        numPoints * 9 * sizeof(float)));


    // =========================================================
    // 6. Camera blocks
    // =========================================================

    CUDA_CHECK(cudaMalloc(
        &gpu.d_Hcc,
        numCameras * 36 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &gpu.d_gc,
        numCameras * 6 * sizeof(float)));

    // 每个 Observation 一个 6x3 的 M
    CUDA_CHECK(cudaMalloc(
        &gpu.d_M,
        numObservations * 18 * sizeof(float)));
}