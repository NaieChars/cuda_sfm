#include "FeatureExtractor.h"
#include <cstring>
#include <iostream>

#include "Config.h"
#include "cudaSift.h"

// OpenCV 版本 SIFT
FeatureSet extractSIFTFeaturesCPU(const cv::Mat& image)
{
	cv::Mat gray;
	if (image.channels() == 3)
	{
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	}
	else
	{
		gray = image;
	}

	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptorsMat;	
	sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptorsMat);

	FeatureSet fs;
	fs.numFeatures = static_cast<int>(keypoints.size());

	if (fs.numFeatures == 0)
	{
		std::cerr << "[FeatureExtractor] Warnig: No feature points were extracted!" << std::endl;
		return fs;
	}

	fs.kpX.resize(fs.numFeatures);
	fs.kpY.resize(fs.numFeatures);
	for (int i = 0; i < fs.numFeatures; i++)
	{
		fs.kpX[i] = keypoints[i].pt.x;
		fs.kpY[i] = keypoints[i].pt.y;
	}

	// flatten the descriptors' matrix
	fs.descriptors.resize(static_cast<size_t>(fs.numFeatures) * FeatureSet::DESC_DIM);	// rsize_t = size_t
	std::memcpy(fs.descriptors.data(), descriptorsMat.ptr<float>(0), fs.descriptors.size() * sizeof(float));

	fs.cvKeypoints = std::move(keypoints);

	return fs;
}


#if USE_CUDA_SIFT

static FeatureSet extractSIFTFeaturesCUDA(const cv::Mat& image)
{
    cv::Mat gray;

    if (image.channels() == 3)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = image;
    }

    // CudaSift 需要 float 灰度图
    cv::Mat floatGray;
    gray.convertTo(floatGray, CV_32FC1);

    int w = floatGray.cols;
    int h = floatGray.rows;


    // 创建 CUDA 图像
    CudaImage cudaImg;

    cudaImg.Allocate(
        w,
        h,
        iAlignUp(w, 128),
        false,
        nullptr,
        reinterpret_cast<float*>(floatGray.data)
    );

    cudaImg.Download();

    // 初始化 SiftData
    SiftData siftData;
    InitSiftData(
        siftData,
        32768,
        true,
        true
    );

    // SIFT 参数
    int numOctaves = 5;
    double initBlur = 1.0;
    float thresh = 2.5f;

    // 临时显存
    float* tempMemory = AllocSiftTempMemory(
        w,
        h,
        numOctaves,
        false
    );

    // 提取 SIFT
    ExtractSift(
        siftData,
        cudaImg,
        numOctaves,
        initBlur,
        thresh,
        0.0f,
        false,
        tempMemory
    );

    FreeSiftTempMemory(tempMemory);

    // 转成你的 FeatureSet
    FeatureSet fs;
    fs.numFeatures = siftData.numPts;

    if (fs.numFeatures == 0)
    {
        std::cerr
            << "[FeatureExtractor] Warning: No feature points were extracted!"
            << std::endl;

        FreeSiftData(siftData);
        return fs;
    }

    fs.kpX.resize(fs.numFeatures);
    fs.kpY.resize(fs.numFeatures);

    fs.descriptors.resize(
        static_cast<size_t>(fs.numFeatures) *
        FeatureSet::DESC_DIM
    );

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(fs.numFeatures);

    SiftPoint* points = siftData.h_data;

    for (int i = 0; i < fs.numFeatures; i++)
    {
        // 关键点坐标
        fs.kpX[i] = points[i].xpos;
        fs.kpY[i] = points[i].ypos;

        // 转换成 OpenCV KeyPoint
        keypoints.emplace_back(
            cv::Point2f(
                points[i].xpos,
                points[i].ypos
            ),
            points[i].scale,
            points[i].orientation
        );

        // 128 维 descriptor
        std::memcpy(
            fs.descriptors.data() +
                i * FeatureSet::DESC_DIM,

            points[i].data,

            FeatureSet::DESC_DIM * sizeof(float)
        );
    }

    fs.cvKeypoints = std::move(keypoints);

    // 释放 CudaSift 数据
    FreeSiftData(siftData);

    return fs;
}

#endif

FeatureSet extractSIFTFeatures(const cv::Mat& image)
{
#if USE_CUDA_SIFT
    return extractSIFTFeaturesCUDA(image);
#else
    return extractSIFTFeaturesCPU(image);
#endif
}

// Read images
static cv::Mat robustImreadIndexed(int index)
{
    std::string filename = "data/img" + std::to_string(index) + ".jpg";
    std::vector<std::string> candidates = 
	{
        filename, "../" + filename, "../../" + filename, "../../../" + filename
    };

    for (const auto& path : candidates) 
	{
        cv::Mat img = cv::imread(path);
        if (!img.empty()) return img;
    }

    return cv::Mat();
}


// ------------------- 外部接口 ---------------------


std::vector<FeatureSet> extractFeaturesFromImages(std::vector<cv::Mat>& images)
{
    std::vector<FeatureSet> results;

	int idx = 1;
	while (true)
	{
		cv::Mat img = robustImreadIndexed(idx);
		if (img.empty()) break;
		images.push_back(img);
		idx++;
	}
	const int n = static_cast<int>(images.size());

	if (n < 2)
	{
		std::cerr << "[FeatureExtractor] Need at least 2 images, found " << n  << std::endl;
		return results;
	}
	std::cout << "[SFM] Found " << n << " images" << std::endl;

	for (int i = 0; i < n; i++)
	{
		FeatureSet feature = extractSIFTFeatures(images[i]);
		std::cout << "[FeatureExtractor] img" << i <<  " Extracted " << feature.numFeatures << " feature points." << std::endl;
		results.push_back(feature);	
	}

	return results;
}



