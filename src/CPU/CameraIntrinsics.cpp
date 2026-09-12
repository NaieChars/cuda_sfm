#include "CameraIntrinsics.h"

CameraIntrinsics createCameraIntrinsics(float fx, float fy, float cx, float cy)
{
    CameraIntrinsics intr;
    intr.fx = fx;
    intr.fy = fy;
    intr.cx = cx;
    intr.cy = cy;

    intr.K = (cv::Mat_<double>(3, 3) <<
        static_cast<double>(intr.fx), 0.0, static_cast<double>(intr.cx),
        0.0, static_cast<double>(intr.fy), static_cast<double>(intr.cy),
        0.0, 0.0, 1.0
    );

    intr.Kinv = intr.K.inv();

    return intr;
}

cv::Vec2f pixelToNormalized(const CameraIntrinsics& intr, float px, float py)
{
	float xn = (px - intr.cx) / intr.fx;
	float yn = (py - intr.cy) / intr.fy;
	return cv::Vec2f(xn, yn);
}