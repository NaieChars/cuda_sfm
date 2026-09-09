#pragma once

__global__ void validateEssentialMatrixCUDA(const double* E, 
     const float* points1, const float* points2, 
     unsigned char* inlierMask, int* inlierCount, int numPoints,
    const float threshold);