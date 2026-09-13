#include "cudaSift.h"
#include "DataStruct.h"
#include "SFM.h"
#include "runBA.cuh"

#include <chrono>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>  // 关闭终端Info
#include <cuda_runtime.h>

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR); // 关闭终端info

    auto start = std::chrono::high_resolution_clock::now();

    // 初始化 CUDA
    InitCuda(0);


    // ---------------- 增量式SFM ------------------
    CameraIntrinsics intr = createCameraIntrinsics(3334.23, 3341.06, 2057.09, 1548.96);
    int imagesProcess;
    SFMResult result = runSFM(intr, imagesProcess);


    // -------------------- BA --------------------
    runBA(result, intr);

    // ----------- 耗时输出 ---------------
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n====================================\n";
    std::cout << "Total execution time: "
              << elapsed
              << " ms\n";
    std::cout << "====================================\n";

    return 0;
}