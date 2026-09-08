#pragma once

__global__ void triangulateKernel(
    const float* points1,
    const float* points2,
    double* P1,
    double* P2,
    int numPoints,
    float* points3D,
    const float reprojThreshold,
    int* Mask);