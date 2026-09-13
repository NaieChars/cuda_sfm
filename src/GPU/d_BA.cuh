#pragma once

#include <cuda_runtime.h>

struct d_Observation
{
    int cameraId;
    int pointId;
    float u;
    float v;
};


struct d_CameraPose
{
    double R[9];
    double t[3];
};


__global__ void computeObsJacobianKernel(
    const d_Observation* observations,
    const d_CameraPose* cameras,
    const float3* points,
    int numObservations,
    float2* residuals,
    float* Jp,
    float* Jc
);

__global__ void computePointBlocksKernel(
    const d_Observation* observations,
    const float2* residuals,
    const float* Jp,
    const int* pointObsStart,
    const int* pointObsCount,
    const int* pointObsIndices,
    int numPoints,
    float* Hpp,
    float* gp
);


__global__ void computeCameraBlocksKernel(
    const d_Observation* observations,
    const float2* residuals,
    const float* Jp,
    const float* Jc,

    const int* cameraObsStart,
    const int* cameraObsCount,
    const int* cameraObsIndices,

    int numCameras,

    float* Hcc,
    float* gc,
    float* M
);


__global__ void invertPointBlocksKernel(
    const float* Hpp,
    float* HppInv,
    int numPoints);



__global__ void computeSchurContributionKernel(
    const d_Observation* observations,
    const float* M,
    const float* HppInv,
    const float* gp,

    const int* pointObsStart,
    const int* pointObsCount,
    const int* pointObsIndices,

    int numPoints,
    int numCameras,

    float* schurH,
    float* schurB);


__global__ void addCameraBlocksToSchurKernel(
    const float* Hcc,
    const float* gc,
    int numCameras,
    float* schurH,
    float* schurB);



__global__ void computePointUpdateKernel(
    const d_Observation* observations,
    const float* M,
    const float* HppInv,
    const float* gp,

    const int* pointObsStart,
    const int* pointObsCount,
    const int* pointObsIndices,

    const float* deltaCamera,

    int numPoints,

    float3* points,
    float step
);