#pragma once

#include <iostream>
#include "CameraIntrinsics.h"
#include "EssentialMatrix.cuh"
#include "FeatureExtractor.h"
#include "FeatureMatch.cuh"
#include "PoseRecovery.h"
#include "Triangulation.cuh"
#include "SFMTypes.h"
#include "util.h"
#include "PnP.h"

#include <unordered_map>
#include <algorithm>
#include <string>

SFMResult runSFM(const CameraIntrinsics& intr, int& imagesProcessed);