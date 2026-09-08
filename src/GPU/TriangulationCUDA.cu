#include "TriangulationCUDA.cuh"
#include "GPU_util.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void triangulateKernel(
    const float* points1,
    const float* points2,
    double* P1,
    double* P2,
    int numPoints,
    float* points3D,
    const float reprojThreshold,
    int* Mask,
    int* flailSolve,
    int* failW,
    int* failDepth1,
    int* failDepth2,
    int* failReproj
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;

    // 读取第 i 个匹配点
    float u1 = points1[2 * i];
    float v1 = points1[2 * i + 1];
    float u2 = points2[2 * i];
    float v2 = points2[2 * i + 1];

    // 构造 A[4][4]
    double A[4][4];
    double* P1_0 = &P1[0];
    double* P1_1 = &P1[4];
    double* P1_2 = &P1[8];
    double* P2_0 = &P2[0];
    double* P2_1 = &P2[4];
    double* P2_2 = &P2[8];

    for (int i = 0; i < 4; i++)
        A[0][i] = v1 * P1_2[i] - P1_1[i];

    for (int i = 0; i < 4; i++)
        A[1][i] = P1_0[i] - u1 * P1_2[i];

    for (int i = 0; i < 4; i++)
        A[2][i] = v2 * P2_2[i] - P2_1[i];
    
    for (int i = 0; i < 4; i++)
        A[3][i] = P2_0[i] - u2 * P2_2[i];

    // 求解 A X = 0 
    double X[4];
    if (!solveAX0(A, X)) {Mask[i] = 0; atomicAdd(flailSolve, 1); return;}
    
    
    
    // 齐次坐标归一化，得到3维点坐标
    double w = X[3];
    if (fabs(w) < 1e-12) {Mask[i] = 0; atomicAdd(failW, 1); return;}
    
    double x = X[0] / w;
    double y = X[1] / w;
    double z = X[2] / w;
    
    

    // 后续用归一化坐标求解
    // 两相机深度检测
    if (z <= 0.0) {Mask[i] = 0; atomicAdd(failDepth1, 1); return;}

    double z2 = P2[8] * x + P2[9] * y + P2[10] * z + P2[11];
    if (z2 <= 0.0) {Mask[i] = 0; atomicAdd(failDepth2, 1); return;}

    // 计算重投影误差
    double P1_2_dot_X = P1[8] * x + P1[9] * y + P1[10] * z + P1[11];
    double u1_proj = (P1[0] * x + P1[1] * y + P1[2] * z + P1[3]) / P1_2_dot_X;
    double v1_proj = (P1[4] * x + P1[5] * y + P1[6] * z + P1[7]) / P1_2_dot_X;

    double P2_2_dot_X = z2;
    double u2_proj = (P2[0] * x + P2[1] * y + P2[2] * z + P2[3]) / P2_2_dot_X;
    double v2_proj = (P2[4] * x + P2[5] * y + P2[6] * z + P2[7]) / P2_2_dot_X;

    double err1 = hypot((double)u1 - u1_proj, (double)v1 - v1_proj);
    double err2 = hypot((double)u2 - u2_proj, (double)v2 - v2_proj);
    double maxErr = max(err1, err2);

    if (maxErr < (double)reprojThreshold)
    {
        points3D[3 * i] = (float)x;
        points3D[3 * i + 1] = (float)y;
        points3D[3 * i + 2] = (float)z;

        Mask[i] = 1;
    }
    else {Mask[i] = 0;atomicAdd(failReproj, 1); return;}
}