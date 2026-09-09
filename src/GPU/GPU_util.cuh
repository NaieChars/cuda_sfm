#pragma once
// 求解 AX = 0 的最小二乘解
__device__ bool solveAX0(const double A[4][4], double X[4]);

__device__ float2 arrayToFloat2(float u, float v);