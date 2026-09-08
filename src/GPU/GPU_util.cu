#include "GPU_util.cuh"

// 求解 AX = 0 的最小二乘解
__device__ bool solveAX0(const double A[4][4], double X[4])
{
    // B = A^T A
    double B[4][4] = {0.0};
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < 4; k++)
            {
                sum += A[k][i] * A[k][j];
            }
            B[i][j] = sum;
        }
    }

    double V[4][4] = {0.0};
    for (int i = 0; i < 4; i++) V[i][i] = 1.0;

    const double eps = 1e-12;
    const int maxIter = 100;       
    int iter = 0;
    while (iter < maxIter) 
    {
        // 寻找最大非对角元
        double maxOff = 0.0;
        int p = 0, q = 1;
        for (int i = 0; i < 4; ++i) 
        {
            for (int j = i + 1; j < 4; ++j) 
            {
                double val = fabs(B[i][j]);
                if (val > maxOff) 
                {
                    maxOff = val;
                    p = i;
                    q = j;
                }
            }
        }
        if (maxOff < eps) break; 

        // 计算旋转角度
        double theta = 0.5 * atan2(2.0 * B[p][q], B[p][p] - B[q][q]);
        double c = cos(theta);
        double s = sin(theta);

        // ---- 对 B 应用雅可比旋转 ----
        double Bpp = B[p][p], Bqq = B[q][q], Bpq = B[p][q];

        B[p][p] = Bpp * c * c + Bqq * s * s - 2.0 * Bpq * s * c;
        B[q][q] = Bpp * s * s + Bqq * c * c + 2.0 * Bpq * s * c;
        B[p][q] = 0.0;
        B[q][p] = 0.0;

        for (int i = 0; i < 4; ++i) 
        {
            if (i != p && i != q) 
            {
                double Bip = B[i][p];
                double Biq = B[i][q];
                B[i][p] = Bip * c - Biq * s;
                B[p][i] = B[i][p];
                B[i][q] = Bip * s + Biq * c;
                B[q][i] = B[i][q];
            }
        }

        // ---- 对 V 应用相同旋转（作用于列） ----
        for (int i = 0; i < 4; ++i) 
        {
            double Vip = V[i][p];
            double Viq = V[i][q];
            V[i][p] = Vip * c - Viq * s;
            V[i][q] = Vip * s + Viq * c;
        }

        ++iter;
    }

    // 找到最小特征值对应的列
    double minEig = B[0][0];
    int minIdx = 0;
    for (int i = 1; i < 4; ++i) 
    {
        if (B[i][i] < minEig) 
        {
            minEig = B[i][i];
            minIdx = i;
        }
    }

    // 提取特征向量并归一化
    double norm = 0.0;
    for (int i = 0; i < 4; ++i) 
    {
        X[i] = V[i][minIdx];
        norm += X[i] * X[i];
    }
    norm = sqrt(norm);
    if (norm < 1e-15) return false;   // 求解失败

    for (int i = 0; i < 4; ++i) X[i] /= norm;
    return true;
}


// 将像素坐标转换成归一化相机坐标，方便 GPU 内的 E 验证
// 返回 float2 ，保存处理后的坐标
__device__ float2 normalizePoints(float u, float v,
    float fx, float fy, float cx, float cy)
{
    return make_float2((u - cx) / fx, (v - cy) / fy);
}