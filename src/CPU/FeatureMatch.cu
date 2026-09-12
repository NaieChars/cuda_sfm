#include <cuda_runtime.h>
#include "../GPU/d_FeatureSet.cuh"
#include "../GPU/d_FeatureMatch.cuh"
#include "cuda_Check.cuh"
#include "DataStruct.h"
#include "FeatureMatch.cuh"


int countValidFeature(const std::vector<cuda_MatchResult>& h_result)
{
    int count = 0;
    for (const auto& result : h_result)
    {
        if (result.bestIdx >= 0)
        count++;
    }
    return count;
}



std::vector<cuda_MatchResult> cuda_FeatureMatch(const FeatureSet& f1, const FeatureSet& f2)
{
    float* d_desc1 = nullptr;
    float* d_desc2 = nullptr;
    cuda_MatchResult* d_result = nullptr;
    const size_t bytes1 = static_cast<size_t>(f1.numFeatures) * 128 * sizeof(float);
    const size_t bytes2 = static_cast<size_t>(f2.numFeatures) * 128 * sizeof(float);
    const size_t resultBytes = static_cast<size_t>(f1.numFeatures) * sizeof(cuda_MatchResult);
    
    CUDA_CHECK(cudaMalloc(&d_desc1, bytes1));
    CUDA_CHECK(cudaMemcpy(d_desc1, f1.descriptors.data(), bytes1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_desc2, bytes2));
    CUDA_CHECK(cudaMemcpy(d_desc2, f2.descriptors.data(), bytes2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_result, resultBytes));
    
    int blockSize = 256;
    int gridSize = (f1.numFeatures + blockSize - 1) / blockSize;
    
    matchBruteForce<<<gridSize, blockSize>>>(d_desc1, f1.numFeatures, d_desc2, f2.numFeatures, 0.75f, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<cuda_MatchResult> h_result(f1.numFeatures);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, resultBytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_desc1));
    CUDA_CHECK(cudaFree(d_desc2));
    CUDA_CHECK(cudaFree(d_result));
    
    return h_result;
}



void convertMatches(const FeatureSet& f1, const FeatureSet& f2, const std::vector<cuda_MatchResult>& cuda_result,
                    std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2,
                    std::vector<int>& indices1, std::vector<int>& indices2)
{
    for (int i = 0; i < f1.numFeatures; i++)
    {
        int j = cuda_result[i].bestIdx;

        if (j < 0)
            continue;
        
        points1.push_back(f1.cvKeypoints[i].pt);
        points2.push_back(f2.cvKeypoints[j].pt);

        indices1.push_back(i);
        indices2.push_back(j);

        // 保存原始索引的原因是经过 bestIdx < 0 的筛选，再 push_back 进 points 后，
        // points 里的点只剩有效点了，原始点的索引消失
    }
}