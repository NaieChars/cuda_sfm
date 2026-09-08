#pragma once
#include "d_FeatureSet.cuh"

__global__ void matchBruteForce(const float* descriptors1, int numFeature1, 
    const float* descriptors2, int numFeature2, float ratioThresh, cuda_MatchResult* results);