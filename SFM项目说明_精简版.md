# SFM 项目说明

## 1. 项目概述

本项目实现一个**双视图 SfM（Structure from Motion）重建流程**，输入两幅具有重叠视野的图像及相机内参，经过特征提取、特征匹配、几何约束、相机姿态恢复和三角化，最终生成稀疏三维点云。

项目采用 **CPU + CUDA GPU** 混合方式：

- CPU / OpenCV：负责 SIFT 特征提取、本质矩阵估计、相机姿态恢复以及流程组织。
- CUDA：负责特征描述子暴力匹配和三角化等计算密集任务。

---

## 2. 整体流程

```text
输入图像 + 相机内参
        │
        ▼
   SIFT 特征提取
        │
        ▼
  CUDA 特征描述子匹配
        │
        ▼
     匹配点坐标
        │
        ▼
 Essential Matrix + RANSAC
        │
        ▼
      恢复 R、t
        │
        ▼
    内点进一步筛选
        │
        ▼
     CUDA 三角化
        │
        ▼
      稀疏 3D 点云
```

主流程由 `runSFM()` 负责串联各模块。

---

## 3. 核心模块

| 模块 | 文件 | 主要职责 | 主要计算位置 |
|---|---|---|---|
| SFM 主流程 | `SFM.cpp/.h` | 串联整个双视图 SfM 流程 | CPU |
| 特征提取 | `FeatureExtractor.cpp/.h` | 读取图像并提取 SIFT 特征 | CPU |
| 特征匹配 | `FeatureMatch.cu/.cuh` | 对 SIFT 描述子进行暴力匹配 | CUDA |
| 本质矩阵 | `EssentialMatrix.cpp/.h` | 使用 RANSAC 估计本质矩阵并得到内点掩码 | CPU / OpenCV |
| 姿态恢复 | `PoseRecovery.cpp/.h` | 根据本质矩阵恢复旋转 `R` 和平移 `t` | CPU / OpenCV |
| 三角化 | `Triangulation.cu/.cuh` | 根据两视图几何关系恢复 3D 点 | CUDA |
| 相机内参 | `CameraIntrinsics.cpp/.h` | 保存相机内参并构造 `K`、`Kinv` | CPU |
| 工具函数 | `util.cpp/.h` | 根据掩码筛选点等 | CPU |
| CUDA 工具函数 | `util.cu/.cuh` | 提供 GPU 端数学计算函数 | CUDA |
| CUDA 数据结构 | `d_FeatureSet.cuh` | 定义 GPU 特征和匹配结果结构 | CUDA |

---

## 4. SFM 主流程

### 核心接口

```cpp
SFMResult runSFM(const CameraIntrinsics& intr);
```

### 流程

```text
extractFeaturesFromImages()
        ↓
 cuda_FeatureMatch()
        ↓
    convertMatches()
        ↓
 estimateEssentialMatrix()
        ↓
   recoverCameraPose()
        ↓
   筛选有效内点
        ↓
 cudaTriangulation()
        ↓
     SFMResult
```

`runSFM()` 最终将三角化得到的 `cv::Point3f` 转换为 `ColoredPoint3D`，写入 `result.pointCloud`。

当前点云颜色在主流程中统一设置为白色。

---

## 5. 特征提取

**文件：** `FeatureExtractor.cpp/.h`

### 核心接口

```cpp
std::vector<FeatureSet> extractFeaturesFromImages();
```

函数从固定命名规则的图像文件中读取图像，并对每张图像执行 SIFT 特征提取。

每张图像对应一个 `FeatureSet`，主要包含：

- 特征点数量 `numFeatures`
- 特征点坐标 `kpX / kpY`
- SIFT 描述子 `descriptors`
- OpenCV 特征点 `cvKeypoints`

SIFT 描述子维度为 128。

输出的 `features` 按图像读取顺序保存，后续用于特征匹配。

---

## 6. CUDA 特征匹配

**文件：** `FeatureMatch.cu/.cuh`

### 核心接口

```cpp
std::vector<cuda_MatchResult>
cuda_FeatureMatch(const FeatureSet& f1, const FeatureSet& f2);
```

其核心 CUDA 核函数为：

```cpp
__global__ void matchBruteForce(...);
```

### 基本过程

```text
左图 SIFT 描述子
        │
        ├── 与右图所有描述子计算距离
        │
        ├── 找到最近邻和次近邻
        │
        └── 进行比值测试
                ↓
            有效匹配
```

每个 CUDA 线程负责一个左图特征点，遍历右图特征描述子，计算平方 L2 距离并进行比值测试。

当前代码中的比值测试阈值为 `0.75f`。

### 匹配结果转换

```cpp
void convertMatches(...);
```

该函数将匹配结果转换为两幅图像中对应的 `cv::Point2f` 像素坐标，分别保存到 `points1` 和 `points2`。

---

## 7. 本质矩阵估计

**文件：** `EssentialMatrix.cpp/.h`

### 核心接口

```cpp
cv::Mat estimateEssentialMatrix(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const CameraIntrinsics& intr,
    std::vector<uchar>& essentialMask
);
```

该模块使用 OpenCV 的 `cv::findEssentialMat()`，结合相机内参和 RANSAC，从匹配点中估计本质矩阵 `E`，并输出 `essentialMask`。

`essentialMask` 用于区分满足当前几何约束的匹配点。

当前代码使用的 RANSAC 参数包括：

- 置信度：`0.999`
- 阈值：`1.5`

---

## 8. 相机姿态恢复

**文件：** `PoseRecovery.cpp/.h`

### 核心接口

```cpp
bool recoverCameraPose(
    const cv::Mat3d& E,
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const CameraIntrinsics& intr,
    const std::vector<uchar>& essentialMask,
    cv::Mat& R,
    cv::Mat& t,
    int& inlierCount,
    std::vector<uchar>& poseMask
);
```

内部首先根据 `essentialMask` 筛选匹配点，然后调用：

```cpp
cv::recoverPose(...);
```

得到：

- `R`：旋转矩阵
- `t`：平移方向
- `poseMask`：姿态恢复阶段使用的掩码
- `inlierCount`：有效内点数量

当前代码以 `inlierCount > 15` 作为姿态是否有效的判断条件。

---

## 9. 内点筛选

**文件：** `util.cpp/.h`

核心函数：

```cpp
std::vector<cv::Point2f> fliterInlierPoints(
    const std::vector<cv::Point2f>& points,
    const std::vector<uchar>& Mask
);
```

作用很简单：根据掩码保留对应位置的点。

在主流程中，匹配点会经历基于几何约束的筛选，随后将有效点用于三角化。

> 注意：原始说明中对 `poseMask` 与经过 `essentialMask` 筛选后的点集之间的索引对应关系存在描述不一致，因此这里不对具体索引实现做额外推断。实际行为应以当前源码为准。

---

## 10. CUDA 三角化

**文件：** `Triangulation.cu/.cuh`

### 核心接口

```cpp
std::vector<cv::Point3f> cudaTriangulation(
    const std::vector<cv::Point2f>& inlierPoints1,
    const std::vector<cv::Point2f>& inlierPoints2,
    const CameraIntrinsics& intr,
    const cv::Mat& R,
    const cv::Mat& t,
    std::vector<int> Mask3D
);
```

三角化使用两幅图像的投影矩阵：

```text
P1 = K [I | 0]
P2 = K [R | t]
```

然后将点坐标和投影矩阵传入 GPU。

### CUDA 核函数

```cpp
__global__ void triangulateKernel(...);
```

每个线程负责一个匹配点对，主要执行：

1. 根据两个投影矩阵构造三角化方程；
2. 求解齐次三维坐标；
3. 进行深度有效性检查；
4. 计算重投影误差；
5. 根据阈值判断该 3D 点是否有效。

当前实现使用的重投影误差阈值为 `1.5`。

最后将 GPU 中得到的有效点回传到 CPU，形成 `std::vector<cv::Point3f>`。

---

## 11. 相机内参

**文件：** `CameraIntrinsics.cpp/.h`

### 数据结构

```cpp
struct CameraIntrinsics
```

主要保存：

- `fx`、`fy`：焦距
- `cx`、`cy`：主点
- `K`：相机内参矩阵
- `Kinv`：内参矩阵的逆

内参矩阵形式为：

```text
[ fx   0  cx ]
[  0  fy  cy ]
[  0   0   1 ]
```

当前代码没有实现畸变模型。

---

## 12. CUDA 数据结构

**文件：** `d_FeatureSet.cuh`

### `cuda_FeatureSet`

保存 GPU 端特征描述子相关数据：

```cpp
struct cuda_FeatureSet {
    int numFeatures;
    float* descriptors;
};
```

### `cuda_MatchResult`

保存单个特征点的匹配结果，当前结构中包含：

```cpp
struct cuda_MatchResult {
    int bestIdx;
    float bestDist;
};
```

---

## 13. CUDA 数学工具

**文件：** `util.cu/.cuh`

主要提供 GPU 端三角化使用的辅助函数。

### `solveAX0`

```cpp
__device__ bool solveAX0(const double A[4][4], double X[4]);
```

用于求解齐次方程：

```text
AX = 0
```

得到三角化所需的齐次三维坐标。

### `normalizePoints`

```cpp
__device__ float2 normalizePoints(
    float u, float v,
    float fx, float fy,
    float cx, float cy
);
```

按照相机内参将像素坐标转换为归一化坐标。

当前说明中，`triangulateKernel` 并未直接调用该函数。

---

## 14. 未完成模块

**文件：** `findEssentialMat.cu/.cuh`

该模块目前没有实际运行功能，文件中只有被注释掉的 CUDA 本质矩阵验证相关代码，因此暂不属于当前 SfM 主流程。

---

## 15. 当前项目结构总结

```text
                   ┌─────────────────────┐
                   │     SFM::runSFM     │
                   └──────────┬──────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     FeatureExtractor                  CameraIntrinsics
              │
              ▼
       SIFT FeatureSet
              │
              ▼
       CUDA FeatureMatch
              │
              ▼
         Match Points
              │
              ▼
       EssentialMatrix
              │
              ▼
        PoseRecovery
          │        │
          │        └── R、t
          ▼
       有效内点
              │
              ▼
       CUDA Triangulation
              │
              ▼
         3D Point Cloud
```

### 当前项目定位

当前实现属于**双视图 SfM**：核心目标是从两幅图像恢复两台相机之间的相对运动，并通过三角化得到稀疏三维点云。

目前的主流程重点集中在：

**SIFT → CUDA 匹配 → 本质矩阵 → 姿态恢复 → CUDA 三角化 → 稀疏点云**

尚未包含增量式 SfM、全局多视图位姿优化以及完整的 Bundle Adjustment 等更高层的多视图重建流程。
