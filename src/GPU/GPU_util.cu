#include "GPU_util.cuh"

// 求解 AX = 0 的最小二乘解
__device__ bool solveAX0(const double A[4][4], double X[4])
{
    // B = A^T A
    double B[4][4] = {0.0};

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 4; ++k)
                B[i][j] += A[k][i] * A[k][j];
        }
    }

    // V：特征向量矩阵
    double V[4][4] = {0.0};
    for (int i = 0; i < 4; ++i)
        V[i][i] = 1.0;

    const double eps = 1e-12;

    for (int iter = 0; iter < 100; ++iter)
    {
        // 找最大的非对角元素
        int p = 0, q = 1;
        double maxOff = fabs(B[0][1]);

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

        if (maxOff < eps)
            break;

        double app = B[p][p];
        double aqq = B[q][q];
        double apq = B[p][q];

        // Jacobi rotation
        double theta = 0.5 * atan2(2.0 * apq, aqq - app);
        double c = cos(theta);
        double s = sin(theta);

        // 更新 B 的非 p/q 元素
        for (int k = 0; k < 4; ++k)
        {
            if (k != p && k != q)
            {
                double bkp = B[k][p];
                double bkq = B[k][q];

                B[k][p] = c * bkp - s * bkq;
                B[p][k] = B[k][p];

                B[k][q] = s * bkp + c * bkq;
                B[q][k] = B[k][q];
            }
        }

        // 更新对角块
        B[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        B[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;

        B[p][q] = 0.0;
        B[q][p] = 0.0;

        // V = V * J
        for (int k = 0; k < 4; ++k)
        {
            double vkp = V[k][p];
            double vkq = V[k][q];

            V[k][p] = c * vkp - s * vkq;
            V[k][q] = s * vkp + c * vkq;
        }
    }

    // 找最小特征值
    int minIdx = 0;
    for (int i = 1; i < 4; ++i)
    {
        if (B[i][i] < B[minIdx][minIdx])
            minIdx = i;
    }

    // 取对应特征向量
    double norm = 0.0;
    for (int i = 0; i < 4; ++i)
    {
        X[i] = V[i][minIdx];
        norm += X[i] * X[i];
    }

    norm = sqrt(norm);

    if (norm < 1e-15)
        return false;

    for (int i = 0; i < 4; ++i)
        X[i] /= norm;

    return true;
}


// 修改：将kernel里的归一化放在了CPU端，此函数退化为单纯返回float2
// 将传入的点坐标数组变为float2形式
__device__ float2 arrayToFloat2(float u, float v)
{
    return make_float2(u, v);
}