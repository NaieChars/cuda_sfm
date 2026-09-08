#pragma once

#include <iostream>
#include "CameraIntrinsics.h"
#include "EssentialMatrix.h"
#include "FeatureExtractor.h"
#include "FeatureMatch.cuh"
#include "PoseRecovery.h"
#include "Triangulation.cuh"
#include "SFMTypes.h"
#include "util.h"

SFMResult runSFM(const CameraIntrinsics& intr);