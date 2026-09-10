#include "FeatureExtractor.h"
#include <cstring>
#include <iostream>

FeatureSet extractSIFTFeatures(const cv::Mat& image)
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

	CV_Assert(descriptorsMat.type() == CV_32F);	
	CV_Assert(descriptorsMat.cols == FeatureSet::DESC_DIM);
	CV_Assert(descriptorsMat.isContinuous());

	fs.kpX.resize(fs.numFeatures);
	fs.kpY.resize(fs.numFeatures);
	for (int i = 0; i < fs.numFeatures; i++)
	{
		fs.kpX[i] = keypoints[i].pt.x;
		fs.kpY[i] = keypoints[i].pt.y;
	}

	// flatten the descriptors' matrix
	fs.descriptors.resize(static_cast<rsize_t>(fs.numFeatures) * FeatureSet::DESC_DIM);	// rsize_t = size_t
	std::memcpy(fs.descriptors.data(), descriptorsMat.ptr<float>(0), fs.descriptors.size() * sizeof(float));

	fs.cvKeypoints = std::move(keypoints);
	std::cout << "[FeatureExtractor] Extracted " << fs.numFeatures << " feature points." << std::endl;

	return fs;
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
		std::cerr << "[SFM] Need at least 2 images, found " << n  << std::endl;
		return results;
	}
	std::cout << "[SFM] Found " << n << " images" << std::endl;

	for (int i = 0; i < n; i++)
	{
		FeatureSet feature = extractSIFTFeatures(images[i]);
		results.push_back(feature);	
	}

	return results;
}