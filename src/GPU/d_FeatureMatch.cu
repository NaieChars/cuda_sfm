#include <float.h>
#include <cuda_runtime.h>
#include "d_FeatureMatch.cuh"

__device__ inline float squaredL2Distance(const float* a, const float* b)
{
	float sum = 0.0f;
	for (int i = 0; i < 128; i++)
	{
		float diff = a[i] - b[i];
		sum += diff * diff;
	}
	return sum;
}


__global__ void matchBruteForce(const float* descriptors1, int numFeature1, const float* descriptors2, int numFeature2, float ratioThresh, cuda_MatchResult* results)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numFeature1) return;

	const float* descA = descriptors1 + static_cast<size_t>(i) * 128;

	float bestDist = FLT_MAX;		
	float secondDist = FLT_MAX;	
	int bestIdx = -1;

	for (int j = 0; j < numFeature2; j++)
	{
		const float* descB = descriptors2 + static_cast<size_t>(j) * 128;
		float d = squaredL2Distance(descA, descB);

		if (d < bestDist)
		{
			secondDist = bestDist;
			bestDist = d;
			bestIdx = j;
		}
		else if (d < secondDist)
			secondDist = d;
	}

	// ratio test
	if (bestIdx >= 0 && bestDist < ratioThresh * ratioThresh * secondDist)
	{
		results[i].bestIdx = bestIdx;
		results[i].bestDist = bestDist;
	}
	else
	{
		results[i].bestDist = 0.0f;
		results[i].bestIdx = -1;
	}
}