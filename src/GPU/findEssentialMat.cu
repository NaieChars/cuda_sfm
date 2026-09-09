#include "findEssentialMat.cuh"
#include "GPU_util.cuh"

// 该函数拿 E 去检查全部 1000 匹配点对
// 输出的 inlierMask/inlierCount 针对的是本次 E 的检查结果，反应这个E的好坏
// 传入的坐标序列必须是已经经过相机坐标归一化的！
__global__ void validateEssentialMatrixCUDA(const double* E, 
     const float* points1, const float* points2, 
     unsigned char* inlierMask, int* inlierCount, int numPoints,
    const float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;

    float2 norm1 = arrayToFloat2(points1[2 * i], points1[2 * i + 1]);
    float2 norm2 = arrayToFloat2(points2[2 * i], points2[2 * i + 1]);

    double3 x1 = make_double3(norm1.x, norm1.y, 1.0);
    double3 x2 = make_double3(norm2.x, norm2.y, 1.0);

    // 下面计算极线的运算是基于 E 按行存储！
    double3 Ex1;
    Ex1.x = E[0] * x1.x + E[1] * x1.y + E[2] * x1.z;
    Ex1.y = E[3] * x1.x + E[4] * x1.y + E[5] * x1.z; 
    Ex1.z = E[6] * x1.x + E[7] * x1.y + E[8] * x1.z;

    // 计算 E^T * x2
    double3 Etx2;
    Etx2.x = E[0] * x2.x + E[3] * x2.y + E[6] * x2.z;
    Etx2.y = E[1] * x2.x + E[4] * x2.y + E[7] * x2.z;
    Etx2.z = E[2] * x2.x + E[5] * x2.y + E[8] * x2.z;

    // 计算 x2^T Ex1
    double error = x2.x * Ex1.x + x2.y * Ex1.y + x2.z * Ex1.z;
    double numerator = error * error;

    // denominator
    double denominator =  Ex1.x * Ex1.x + Ex1.y * Ex1.y + Etx2.x * Etx2.x + Etx2.y * Etx2.y;

    // Sampson distance
    double sampsonDistance = numerator / (denominator + 1e-12);

    if (sampsonDistance < threshold)
    {
        inlierMask[i] = 1;
        atomicAdd(inlierCount, 1);
    }
    else
    {
        inlierMask[i] = 0;
    }
}