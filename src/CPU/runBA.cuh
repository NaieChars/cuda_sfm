#pragma once

struct SFMResult;
struct CameraIntrinsics;

void runBA(
    SFMResult& result,
    const CameraIntrinsics& intr
);