# SFM 项目说明

## 1. 项目概述

本项目实现一个**增量式 SfM（Structure from Motion）重建流程**，目前处于双视图状态，输入两幅具有重叠视野的图像及相机内参，经过特征提取、特征匹配、几何约束、相机姿态恢复和三角化，最终生成稀疏三维点云。

项目采用 **CPU + CUDA GPU** 混合方式：

- CPU / OpenCV：负责 SIFT 特征提取、本质矩阵估计、相机姿态恢复以及流程组织。
- CUDA：负责特征描述子暴力匹配、三角化、RANSAC验证。

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
 Essential Matrix + CUDA RANSAC
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
| 特征匹配 | `d_FeatureMatch.cu/.cuh` | 对 SIFT 描述子进行暴力匹配 | CUDA |
| 本质矩阵 | `EssentialMatrix.cu/.cuh` `findEssentialMat.cu/cuh` | CPU端获得试验E，传入GPU进行RANSAC验证，最终输出E与掩码 | CPU / GPU / OpenCV |
| 姿态恢复 | `PoseRecovery.cpp/.h` | 根据本质矩阵恢复旋转 `R` 和平移 `t` | CPU / OpenCV |
| 三角化 | `Triangulation.cu/.cuh` | 根据两视图几何关系恢复 3D 点 | CUDA |
| 相机内参 | `CameraIntrinsics.cpp/.h` | 保存相机内参并构造 `K`、`Kinv` | CPU |
| 工具函数 | `util.cpp/.h` | 根据掩码筛选点，匹配点坐标归一化，拍平矩阵 | CPU |
| CUDA 工具函数 | `GPU_util.cu/.cuh` | 提供 GPU 端数学计算函数 | CUDA |
| CUDA 数据结构 | `d_FeatureSet.cuh` | 定义 GPU 特征和匹配结果结构 | CUDA |
| CPU 数据结构 | `FeatureSet.h` `SFMTypes.h` | 定义SIFT出来的CPU特征并拍平传给GPU, 打包SFM的输出 | CPU |

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


## 数据流动说明

- opencv读取图像信息，送入 SIFT 进行特征点提取。提取后的数据打包成结构 `std::vector<FeatuereSet>`，每个 `FeatureSet` 对应一个图像信息
- 两个 `FeatureSet` 进入特征点匹配阶段，输出第一张图的 `cuda_MatchResult`，再基于该匹配结果筛选出已成功匹配的两点，并将他们坐标一一对应保存进 `std::vector<cv::Points2f> points`
- 将 `points1`、`points2` 输入进本质矩阵估计函数，输出本质矩阵E与 `essentialMask`（**该掩码表示这对匹配点是否符合本质矩阵的几何约束**）
- 将 `points1`、`points2` 输入位姿恢复函数，内部先用 `essentialMask` 进行一次筛选，再将筛选后的点进行位姿恢复。位姿恢复输出有效内点数量，`poseMask`，R，t
- 在 SFM 主流程中进行两次点筛选，将其传入三角化函数，输出最终内点集与 3DMask

---



## 5. 特征提取

**文件：** `FeatureExtractor.cpp/.h`

### 核心接口

```cpp
std::vector<FeatureSet> extractFeaturesFromImages();
```

### 内部核心函数

```cpp
static cv::Mat robustImreadIndexed(int index)        // 多路径尝试读图
FeatureSet extractSIFTFeatures(const cv::Mat& image) // 传入图像，进行 SIFT
```

函数从固定命名规则的图像文件中读取图像，并对每张图像执行 SIFT 特征提取。

每张图像对应一个 `FeatureSet`，主要包含：

- 特征点数量 `numFeatures`
- 特征点坐标 `std::vector<float> kpX / std::vector<float> kpY`
- SIFT 描述子 `std::vector<float> descriptors`
- OpenCV 特征点 `std::vector<cv::KeyPoint> cvKeypoints`

其中 `descriptors` 和 `cvKeypoints` 均来自于 SIFT

SIFT 描述子维度为 128。

输出的 `features` 按图像读取顺序保存，后续用于特征匹配。

---

## 6. CUDA 特征匹配

### Host 端
**文件：** `CPU/FeatureMatch.cu/.cuh`

####  Host 端核心接口

```cpp
std::vector<cuda_MatchResult>
cuda_FeatureMatch(const FeatureSet& f1, const FeatureSet& f2);
```

将 Host 端的两个**描述子**拍平为数组，送入 Device 端进行特征点暴力匹配。Device 端输出给 Hoxst 端类型为 `std::vector<cuda_MatchResult>` 的数据结构。

```cpp
void convertMatches(const FeatureSet& f1, const FeatureSet& f2, 
                    const std::vector<cuda_MatchResult>& cuda_result,
                    std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2)
```

匹配结果转换：  
该函数将匹配结果 `cuda_MatchResult` 转换为两幅图像中对应的 `cv::Point2f` 像素坐标，分别保存到 `points1` 和 `points2`。（`points1[i]` 与 `points2[i]` 表示同一个匹配关系）。

此处**经历了一次筛选**，如果第一张图中的某特征点没有在第二张图里找到最佳匹配点，该点直接舍弃。


#### 核心 CUDA 核函数
**文件：`GPU/d_FeatureMatch.cu/cuh`**

```cpp
__global__ void matchBruteForce(...);
```

**基本过程**

暴力匹配：   
左图 SIFT 描述子，与右图所有描述子计算距离，找到最近邻和次近邻，进行比值测试，最终输出有效匹配。            

每个 CUDA 线程负责一个左图特征点，**遍历**右图特征描述子，计算平方 L2 距离并进行比值测试。

当前代码中的比值测试阈值为 `0.75f`。

---

## 7.1 本质矩阵估计_OpenCV

**文件：** `EssentialMatrix.cu/.cuh`

### 核心接口

```cpp
cv::Mat estimateEssentialMatrix(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const CameraIntrinsics& intr,
    std::vector<uchar>& essentialMask
);
```

该模块内部使用 OpenCV 的 `cv::findEssentialMat()`，结合相机内参和 RANSAC，从匹配点中估计本质矩阵 `E`，并输出 `essentialMask`。

`essentialMask` 用于区分满足当前几何约束的匹配点。

当前代码使用的 RANSAC 参数包括：

- 置信度：`0.999`
- 阈值：`1.5`

---

## 7.2 本质矩阵估计_CUDA

**文件：**`findEssentialMat.cu/cuh` `EssentialMatrix.cu/cuh`

### 核心接口

```cpp
// CUDA 版本求E的对外上层接口，对输出结果暂时不包装成 RansacResult
cv::Mat estimateEssentialMatrixRANSAC(const std::vector<cv::Point2f>& points1,
                        const std::vector<cv::Point2f>& points2,
                        double fx, double fy, double cx, double cy, 
                        const float confidence,
                        const int maxIterations,
                        std::vector<uchar>& essentialMask);
```

该模块采用 CPU + CUDA 协同的 RANSAC 方案估计本质矩阵 E。

- 在函数内部，首先对点坐标归一化到相机坐标系。随后在CPU端进行 RANSAC 采样，每次8点，生成本质矩阵。
- 候选E传入GPU，进行对全部匹配点并行计算 Sampson distance，判断匹配点是否为内点，统计当前E的内点数量
- RANSAC 根据内点数选择最佳E，并根据当前最佳内点比例动态调整迭代次数
- RANSAC结束后，使用最佳内点重新拟合E，并再次通过CUDA验证得到essentialMask

当前实现中的主要 RANSAC 参数由接口传入，包括：
- 置信度：0.999
- 最大迭代次数：1000
- Sampson distance 阈值：当前实现为 1e-6

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

内部**首先根据 `essentialMask` 筛选匹配点**，然后调用 OpenCV 提供的函数：

```cpp
cv::recoverPose(...);
```

得到：

- `R`：旋转矩阵
- `t`：平移方向
- **`poseMask`：姿态恢复阶段使用的掩码**
- `inlierCount`：经 `essentialMask` 与位姿恢复筛选后的有效内点数量，其值等于 `poseMask` 后的点数量

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

在主流程中，用于三角化的点会经过 `PoseMask`、`essentialMask` 两次筛选

---

## 10. CUDA 三角化

**文件：** `Triangulation.cu/.cuh`

### 核心接口

```cpp
std::vector<cv::Point3f> cudaTriangulation(
    const std::vector<cv::Point2f>& inlierPoints1,
    const std::vector<cv::Point2f>& inlierPoints2,
    const CameraIntrinsics& intr,
    const CameraPose& pose1,
    const CameraPose& pose2,
    std::vector<int> Mask3D
);
```

三角化使用两幅图像的投影矩阵：

```text
P1 = K [I | 0]
P2 = K [R | t]
```

传入两次筛选后的内点，两次相机的位姿、`CameraIntrinsics`，内部构造两个函数的投影矩阵，将其拍平，连同内点传入进 Device 端进行三角化，Device 端输出 `d_Mask`（即判断该3D点是否有效）到 Host 端。输出 `Mask_3D` 与经过该掩码筛选出的最终内点集

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

当前实现使用的重投影误差阈值为 `1.5`

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

## 12. 增量式 SFM 全流程数据流通

**主要有这四种数据编号**
|`kpPrev`/`kpCur`| `int` | 某张图中的 SIFT 特征编号 |
| `newMatches[k]`| `NewMatch`| 一对匹配的两个点 SIFT 编号|
|`newPointsPrev[k]`|`cv::Point2f`|这对匹配中，上一张图的像素坐标|
|`Point3D[k]`|`cv::Point3f`|这对匹配三角化出来的三维坐标|

1. 原始数据：  
`std::vector<cv::Mat> iamges`  

1. SIFT 
经 SIFT，每一个图片都会获得一个 `featureSet` 数据结构，我们将其打包成 `std::vector<feautreSet> feature`，例如 `feature[0]` 包含第一张图片经 SIFT 获得的数据。一个 `featureSet` 包含当前图像获得的特征点数量，按顺序排列的坐标数组，按顺序排列的描述子数组。例如：

```
features[0]:

feature 0 -> (123, 456) + 126维的descriptor
feature 1 -> ...
```

**featureIndex**：例如 `feature 233` 表示这个图片中233哥SIFT特征点，用于 track 编号

3. 特征匹配
将两张图进行 cuda 匹配：`cuda_FeatureMatch(features[prevIdx], features[i]);`，得到 `std::vector<cuda_MatchResult> matchesPrev;`

```
struct cuda_MatchResult
{
    int bestIdx;
    float bestDist;
};
```

`matchesPrev[kpPrev]` 表示：`prevIdx` 图片的第 `kpPrev` 个特征，匹配到了当前图片的第 `bestIdx` 个特征。

4. 数据转换
`convertMatches()` 把 CUDA 匹配结果转换成SFM使用的数据，输出四个一一对应的数组：

```
points1[k]
points2[k]
indices1[k]
indices2[k]
```

例如：

```
k = 100

points1[100]   = (1234, 567)
points2[100]   = (1301, 580)

indices1[100]  = 532
indices2[100]  = 817
```
表示前一张图的 feature 532 匹配当前图的 feature 817，对应像素 (1234,567) ↔ (1301,580)

5. 本质矩阵估计
上面四个数组经 `essentialMask` 筛选得到第一轮内点

6. 位姿恢复
上面四个数组经 `poseMask` 筛选得到第二轮内点  
同时该过程还会将得到的R，t进行保存，用于下面的三角化

7. 三角化
将经过两次筛选的 `points1` 和 `points2` 输入三角化函数，得到 `Points3D`、`Mask3D`

8. 加入 `pointCloud`
对于有效的 `Points3D[k]`，会加入 `pointCloud`，例如


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
