#include "CameraIntrinsics.h"

CameraIntrinsics createCameraIntrinsics(float fx, float fy, float cx, float cy)
{
    CameraIntrinsics camera;
    camera.fx = fx;
    camera.fy = fy;
    camera.cx = cx;
    camera.cy = cy;

    camera.K = (cv::Mat_<double>(3, 3) <<
        static_cast<double>(camera.fx), 0.0, static_cast<double>(camera.cx),
        0.0, static_cast<double>(camera.fy), static_cast<double>(camera.cy),
        0.0, 0.0, 1.0
    );

    camera.Kinv = camera.K.inv();

    return camera;
}

cv::Vec2f pixelToNormalized(const CameraIntrinsics& intr, float px, float py)
{
	float xn = (px - intr.cx) / intr.fx;
	float yn = (py - intr.cy) / intr.fy;
	return cv::Vec2f(xn, yn);
}