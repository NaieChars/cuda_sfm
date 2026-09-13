#include "d_BA.cuh"


// uv 是归一化后的坐标
__global__ void computeObsJacobianKernel(
    const d_Observation* observations,
    const d_CameraPose* cameras,
    const float3* points,
    int numObservations,
    float2* residuals,
    float* Jp,
    float* Jc
)
{
    int obsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (obsIdx >= numObservations) return;

    d_Observation obs = observations[obsIdx];
    d_CameraPose cam = cameras[obs.cameraId];

    float3 X = points[obs.pointId];

    // Xc 相机坐标系
    float3 Xc;
    Xc.x = cam.R[0] * X.x + cam.R[1] * X.y + cam.R[2] * X.z + cam.t[0];
    Xc.y = cam.R[3] * X.x + cam.R[4] * X.y + cam.R[5] * X.z + cam.t[1];
    Xc.z = cam.R[6] * X.x + cam.R[7] * X.y + cam.R[8] * X.z + cam.t[2];

    // 深度保护
    if (Xc.z <= 1e-6f) 
    {
        residuals[obsIdx] = make_float2(0.f, 0.f);
        float* Jp_o = Jp + obsIdx * 6;
        Jp_o[0]=Jp_o[1]=Jp_o[2]=Jp_o[3]=Jp_o[4]=Jp_o[5]=0.f;

        float* Jc_o = Jc + obsIdx * 12;
        for (int k = 0; k < 12; ++k) Jc_o[k] = 0.f;

        return;
    }

    // 投影
    float2 pX;
    float invZ = 1.f / Xc.z;
    pX.x = Xc.x * invZ;
    pX.y = Xc.y * invZ;

    // Residual
    residuals[obsIdx].x = obs.u - pX.x;
    residuals[obsIdx].y = obs.v - pX.y;

    // Jp
    float* Jp_o = Jp + obsIdx * 6;

    Jp_o[0] = ( -cam.R[0] + pX.x * cam.R[6] ) * invZ;
    Jp_o[1] = ( -cam.R[1] + pX.x * cam.R[7] ) * invZ;
    Jp_o[2] = ( -cam.R[2] + pX.x * cam.R[8] ) * invZ;
    Jp_o[3] = ( -cam.R[3] + pX.y * cam.R[6] ) * invZ;
    Jp_o[4] = ( -cam.R[4] + pX.y * cam.R[7] ) * invZ;
    Jp_o[5] = ( -cam.R[5] + pX.y * cam.R[8] ) * invZ;

    // Jc
    float* Jc_o = Jc + obsIdx * 12;

    float Xrx = Xc.x - cam.t[0];
    float Xry = Xc.y - cam.t[1];
    float Xrz = Xc.z - cam.t[2];

    float invZ2 = invZ * invZ;

    Jc_o[0]  =  pX.x * Xry * invZ2;                                  
    Jc_o[1]  = -Xrz * invZ - pX.x * Xrx * invZ2;                  
    Jc_o[2]  =  Xry * invZ;                                         
    Jc_o[3]  = -invZ;                                             
    Jc_o[4]  =  0.f;                                             
    Jc_o[5]  =  pX.x * invZ2;                                      
    Jc_o[6]  =  Xrz * invZ + pX.y * Xry * invZ2;                  
    Jc_o[7]  = -pX.y * Xrx * invZ2;                               
    Jc_o[8]  = -Xrx * invZ;                                        
    Jc_o[9]  =  0.f;                                          
    Jc_o[10] = -invZ;                                          
    Jc_o[11] =  pX.y * invZ2;
}



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
)
{
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx >= numPoints) return;

    float H[9] = {0.f};
    float g[3] = {0.f};

    int start = pointObsStart[pointIdx];
    int count = pointObsCount[pointIdx];

    for (int k = 0; k < count; k++)
    {
        // 这里假设观测按 pointId 连续排列，obsIdx = start + k
        int obsIdx = pointObsIndices[start + k];

        const float* Jp_o = Jp + obsIdx * 6;   // 2x3 行优先
        float2 r = residuals[obsIdx];

        // Hpp += Jp^T * Jp   (3x3)
        // Jp^T 是 3x2，Jp 是 2x3，乘出来 3x3
        // Jp 第 0 行是 Jp_o[0..2]，第 1 行是 Jp_o[3..5]
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                H[i*3 + j] += Jp_o[0 + i] * Jp_o[0 + j]   // 第 0 行贡献
                            + Jp_o[3 + i] * Jp_o[3 + j];  // 第 1 行贡献
            }
        }

        // gp += Jp^T * r   (3)
        for (int i = 0; i < 3; ++i)
        {
            g[i] += Jp_o[0 + i] * r.x
                  + Jp_o[3 + i] * r.y;
        }
    }

    // 写回
    float* Hpp_o = Hpp + pointIdx * 9;
    float* gp_o  = gp  + pointIdx * 3;

    #pragma unroll
    for (int i = 0; i < 9; ++i) Hpp_o[i] = H[i];

    #pragma unroll
    for (int i = 0; i < 3; ++i) gp_o[i]  = g[i];
}




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
)
{
    int cameraIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cameraIdx >= numCameras)
        return;

    // ------------------------------------------------------------
    // 当前 Camera 的局部 Hcc 和 gc
    //
    // Hcc : 6x6
    // gc  : 6
    // ------------------------------------------------------------

    float H[36] = {0.f};
    float g[6] = {0.f};

    int start = cameraObsStart[cameraIdx];
    int count = cameraObsCount[cameraIdx];

    // ------------------------------------------------------------
    // 遍历当前 Camera 的所有 observation
    // ------------------------------------------------------------

    for (int k = 0; k < count; k++)
    {
        int obsIdx = start + k;

        const float* Jc_o = Jc + obsIdx * 12;
        const float* Jp_o = Jp + obsIdx * 6;

        float2 r = residuals[obsIdx];

        // --------------------------------------------------------
        // Jc 是 2x6：
        //
        // [ Jc[0]  Jc[1]  ... Jc[5]  ]
        // [ Jc[6]  Jc[7]  ... Jc[11] ]
        //
        // Hcc += Jc^T * Jc
        // --------------------------------------------------------

        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                H[i * 6 + j] +=
                    Jc_o[0 * 6 + i] * Jc_o[0 * 6 + j]
                  + Jc_o[1 * 6 + i] * Jc_o[1 * 6 + j];
            }
        }

        // --------------------------------------------------------
        // gc += Jc^T * r
        // --------------------------------------------------------

        for (int i = 0; i < 6; i++)
        {
            g[i] +=
                Jc_o[0 * 6 + i] * r.x
              + Jc_o[1 * 6 + i] * r.y;
        }

        // --------------------------------------------------------
        // M = Jc^T * Jp
        //
        // Jc : 2x6
        // Jp : 2x3
        //
        // M : 6x3
        //
        // 注意：
        // M 不能像 Hcc / gc 一样累加！
        //
        // 每一个 observation 都保存自己的 M。
        // --------------------------------------------------------

        float* M_o = M + obsIdx * 18;

        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                M_o[i * 3 + j] =
                    Jc_o[0 * 6 + i] * Jp_o[0 * 3 + j]
                  + Jc_o[1 * 6 + i] * Jp_o[1 * 3 + j];
            }
        }
    }

    // ------------------------------------------------------------
    // 写回 Hcc
    // ------------------------------------------------------------

    float* Hcc_o = Hcc + cameraIdx * 36;

    #pragma unroll
    for (int i = 0; i < 36; i++)
    {
        Hcc_o[i] = H[i];
    }

    // ------------------------------------------------------------
    // 写回 gc
    // ------------------------------------------------------------

    float* gc_o = gc + cameraIdx * 6;

    #pragma unroll
    for (int i = 0; i < 6; i++)
    {
        gc_o[i] = g[i];
    }
}


__device__ bool invert3x3(
    const float* A,
    float* invA)
{
    float a = A[0];
    float b = A[1];
    float c = A[2];

    float d = A[3];
    float e = A[4];
    float f = A[5];

    float g = A[6];
    float h = A[7];
    float i = A[8];

    // 行列式
    float det =
          a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g);

    // 奇异矩阵
    if (fabsf(det) < 1e-10f)
        return false;

    float invDet = 1.0f / det;

    invA[0] =  (e * i - f * h) * invDet;
    invA[1] =  (c * h - b * i) * invDet;
    invA[2] =  (b * f - c * e) * invDet;

    invA[3] =  (f * g - d * i) * invDet;
    invA[4] =  (a * i - c * g) * invDet;
    invA[5] =  (c * d - a * f) * invDet;

    invA[6] =  (d * h - e * g) * invDet;
    invA[7] =  (b * g - a * h) * invDet;
    invA[8] =  (a * e - b * d) * invDet;

    return true;
}


__global__ void invertPointBlocksKernel(
    const float* Hpp,
    float* HppInv,
    int numPoints)
{
    int pointIdx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (pointIdx >= numPoints)
        return;

    const float* H = Hpp + pointIdx * 9;
    float* invH = HppInv + pointIdx * 9;

    if (!invert3x3(H, invH))
    {
        // 奇异情况下先全部置零
        for (int i = 0; i < 9; i++)
            invH[i] = 0.0f;
    }
}



__device__ void computeSchurBlock(
    const float* Mi,
    const float* Hinv,
    const float* Mj,
    float* result)
{
    // 临时矩阵：
    // temp = Mi * Hinv
    //
    // Mi   : 6x3
    // Hinv : 3x3
    // temp : 6x3

    float temp[18] = {0.f};

    for (int i = 0; i < 6; i++)
    {
        for (int k = 0; k < 3; k++)
        {
            float sum = 0.f;

            for (int l = 0; l < 3; l++)
            {
                sum += Mi[i * 3 + l]
                     * Hinv[l * 3 + k];
            }

            temp[i * 3 + k] = sum;
        }
    }

    // result = temp * Mj^T
    //
    // temp : 6x3
    // Mj^T : 3x6
    // result: 6x6

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            float sum = 0.f;

            for (int k = 0; k < 3; k++)
            {
                sum += temp[i * 3 + k]
                     * Mj[j * 3 + k];
            }

            result[i * 6 + j] = sum;
        }
    }
}



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
    float* schurB)
{
    int pointIdx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (pointIdx >= numPoints)
        return;

    int start = pointObsStart[pointIdx];
    int count = pointObsCount[pointIdx];

    const float* invH =
        HppInv + pointIdx * 9;

    const float* gp_o =
        gp + pointIdx * 3;


    // --------------------------------------------------------
    // Hpp^-1 * gp
    // --------------------------------------------------------

    float invH_gp[3];

    for (int i = 0; i < 3; ++i)
    {
        invH_gp[i] = 0.0f;

        for (int j = 0; j < 3; ++j)
        {
            invH_gp[i] +=
                invH[i * 3 + j] * gp_o[j];
        }
    }


    // --------------------------------------------------------
    // 遍历这个 Point 的所有 Observation
    // --------------------------------------------------------

    for (int a = 0; a < count; ++a)
    {
        int obsA =
            pointObsIndices[start + a];

        int cameraA =
            observations[obsA].cameraId;

        const float* M_A =
            M + obsA * 18;


        // ====================================================
        // b = gc - M Hpp^-1 gp
        // ====================================================

        float M_invHgp[6] = {0.0f};

        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                M_invHgp[i] +=
                    M_A[i * 3 + j] *
                    invH_gp[j];
            }
        }

        for (int i = 0; i < 6; ++i)
        {
            atomicAdd(
                &schurB[cameraA * 6 + i],
                -M_invHgp[i]
            );
        }


        // ====================================================
        // S = Hcc - M Hpp^-1 M^T
        // ====================================================

        for (int b = 0; b < count; ++b)
        {
            int obsB =
                pointObsIndices[start + b];

            int cameraB =
                observations[obsB].cameraId;

            const float* M_B =
                M + obsB * 18;


            // temp = M_A * Hpp^-1
            float temp[18] = {0.0f};

            for (int i = 0; i < 6; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        temp[i * 3 + j] +=
                            M_A[i * 3 + k] *
                            invH[k * 3 + j];
                    }
                }
            }


            // temp * M_B^T
            for (int i = 0; i < 6; ++i)
            {
                for (int j = 0; j < 6; ++j)
                {
                    float value = 0.0f;

                    for (int k = 0; k < 3; ++k)
                    {
                        value +=
                            temp[i * 3 + k] *
                            M_B[j * 3 + k];
                    }

                    atomicAdd(
                        &schurH[
                            (cameraA * 6 + i) *
                            (numCameras * 6) +
                            (cameraB * 6 + j)
                        ],
                        -value
                    );
                }
            }
        }
    }
}



__global__ void addCameraBlocksToSchurKernel(
    const float* Hcc,
    const float* gc,
    int numCameras,
    float* schurH,
    float* schurB)
{
    int cameraIdx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (cameraIdx >= numCameras)
        return;

    // Hcc: 6x6
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            int globalRow = cameraIdx * 6 + i;
            int globalCol = cameraIdx * 6 + j;

            schurH[
                globalRow * (numCameras * 6) +
                globalCol
            ] += Hcc[cameraIdx * 36 + i * 6 + j];
        }
    }

    // gc: 6
    for (int i = 0; i < 6; ++i)
    {
        schurB[cameraIdx * 6 + i] +=
            gc[cameraIdx * 6 + i];
    }
}


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
)
{
    int pointIdx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (pointIdx >= numPoints)
        return;

    // --------------------------------------------------------
    // 当前 Point 的 Hpp^-1
    // --------------------------------------------------------

    const float* invH =
        HppInv + pointIdx * 9;

    const float* gp_o =
        gp + pointIdx * 3;

    // --------------------------------------------------------
    // rhs = -gp
    // --------------------------------------------------------

    float rhs[3];

    for (int i = 0; i < 3; ++i)
    {
        rhs[i] = -gp_o[i];
    }

    // --------------------------------------------------------
    // rhs -= M^T * deltaCamera
    //
    // M : 6x3
    // deltaCamera : 6
    //
    // M^T * deltaCamera : 3
    // --------------------------------------------------------

    int start = pointObsStart[pointIdx];
    int count = pointObsCount[pointIdx];

    for (int k = 0; k < count; ++k)
    {
        int obsIdx =
            pointObsIndices[start + k];

        int cameraIdx =
            observations[obsIdx].cameraId;

        const float* M_o =
            M + obsIdx * 18;

        const float* deltaC =
            deltaCamera + cameraIdx * 6;

        for (int j = 0; j < 3; ++j)
        {
            float sum = 0.0f;

            for (int i = 0; i < 6; ++i)
            {
                sum +=
                    M_o[i * 3 + j] *
                    deltaC[i];
            }

            rhs[j] -= sum;
        }
    }

    // --------------------------------------------------------
    // deltaPoint = Hpp^-1 * rhs
    // --------------------------------------------------------

    float deltaP[3] = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            deltaP[i] +=
                invH[i * 3 + j] *
                rhs[j];
        }
    }

    // --------------------------------------------------------
    // Update Point
    // --------------------------------------------------------

    float3 P = points[pointIdx];

    P.x += step * deltaP[0];
    P.y += step * deltaP[1];
    P.z += step * deltaP[2];

    points[pointIdx] = P;
}