#pragma once

struct cuda_FeatureSet
{
	int numFeatures;
	float* descriptors = nullptr;
};

struct cuda_MatchResult
{
    int bestIdx;
    float bestDist;
};