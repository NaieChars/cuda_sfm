#pragma once

#include <vector>

#include "../GPU/d_BA.cuh"
#include "DataStruct.h"
#include "cuda_Check.cuh"

void buildPointObservationIndex(
    const std::vector<d_Observation>& observations,
    int numPoints,
    std::vector<int>& pointObsStart,
    std::vector<int>& pointObsCount,
    std::vector<int>& pointObsIndices);


void buildCameraObservationIndex(
    const std::vector<d_Observation>& observations,
    int numCameras,
    std::vector<int>& cameraObsStart,
    std::vector<int>& cameraObsCount,
    std::vector<int>& cameraObsIndices);


void uploadBADataToGPU(
    const std::vector<d_Observation>& observations,
    const std::vector<d_CameraPose>& cameras,
    const std::vector<cv::Point3f>& points,
    const std::vector<int>& pointObsStart,
    const std::vector<int>& pointObsCount,
    const std::vector<int>& pointObsIndices,
    const std::vector<int>& cameraObsStart,
    const std::vector<int>& cameraObsCount,
    const std::vector<int>& cameraObsIndices,
    int numObservations,
    int numCameras,
    int numPoints,
    BAGPUData& gpu);