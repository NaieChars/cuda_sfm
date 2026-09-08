#include <iostream>
#include "SFM.h"
#include <opencv2/core/utils/logger.hpp>  // 关闭终端Info

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR); // 关闭终端info

    CameraIntrinsics intr = createCameraIntrinsics(3334.23, 3341.06, 2057.09, 1548.96);

    SFMResult result = runSFM(intr);

    // 打印第二个相机位姿
    if (result.cameras.size() >= 2)
    {
        std::cout << "\nCamera 1 R:" << std::endl;
        std::cout << result.cameras[1].R << std::endl;

        std::cout << "\nCamera 1 t:" << std::endl;
        std::cout << result.cameras[1].t << std::endl;
    }

    return 0;
}