#include <iostream>
#include "SFM.h"
#include <opencv2/core/utils/logger.hpp>  // 关闭终端Info
#include "cudaSift.h"

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR); // 关闭终端info

    // 初始化 CUDA
    InitCuda(0);

    CameraIntrinsics intr = createCameraIntrinsics(3334.23, 3341.06, 2057.09, 1548.96);
    int imagesProcess;
    SFMResult result = runSFM(intr, imagesProcess);

    return 0;
}