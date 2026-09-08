#include "findEssentialMat.cuh"
#include "GPU_util.cuh"

/*
//  直接往GPU里面传E，先将 convertMatches 的 points 转换成相机坐标系，并归一化
__global__ void validateEssentialMatrixCUDA(const double* E, 
     const float* points1, const float* points2, 
     int* inlierMask, int numPoints, int* inlierCount,
     float fx, float fy, float cx, float cy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;

    float2 norm1 = normalizePoints(points1[2 * i], points2[2 * i + 1], fx, fy, cx, cy);
    float2 norm2 = normalizePoints(points2[2 * i], points2[2 * i + 1], fx, fy, cx, cy);

    // 后续完成，可能会有三个函数来重构findEssentialMat
    // CPU生成E，GPU内进行验证，CPU选择最佳E，重复N次
}
    */