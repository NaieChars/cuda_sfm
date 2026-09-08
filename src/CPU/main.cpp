#include <iostream>
#include "SFM.h"
#include <opencv2/core/utils/logger.hpp>  // 关闭终端Info

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR); // 关闭终端info

    CameraIntrinsics intr = createCameraIntrinsics(3334.23, 3341.06, 2057.09, 1548.96);

    SFMResult result = runSFM(intr);

    // ==============================
    // 3. 检查结果
    // ==============================
    std::cout << "========== SfM Finished ==========" << std::endl;

    std::cout << "Camera count: "
              << result.cameras.size() << std::endl;

    std::cout << "3D point count: "
              << result.pointCloud.size() << std::endl;


    // ==============================
    // 4. 打印第二个相机位姿
    // ==============================
    if (result.cameras.size() >= 2)
    {
        std::cout << "\nCamera 1 R:" << std::endl;
        std::cout << result.cameras[1].R << std::endl;

        std::cout << "\nCamera 1 t:" << std::endl;
        std::cout << result.cameras[1].t << std::endl;
    }


    // ==============================
    // 5. 打印前几个 3D 点
    // ==============================
    const int printCount =
        std::min(10, static_cast<int>(result.pointCloud.size()));

    std::cout << "\nFirst " << printCount
              << " 3D points:" << std::endl;

    for (int i = 0; i < printCount; ++i)
    {
        const auto& p = result.pointCloud[i];

        std::cout
            << i << ": ("
            << p.position.x << ", "
            << p.position.y << ", "
            << p.position.z << ")"
            << std::endl;
    }

    return 0;
}