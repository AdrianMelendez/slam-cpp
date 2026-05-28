#pragma once
#include "io/tum_loader.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/features2d.hpp"
#include <vector>

struct Features {
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  cv::Mat depth_map; // float32 meters, empty for monocular frames
};

class FeatureDetector {
public:
  explicit FeatureDetector(int n_features = 1000);
  Features detect(const Frame& frame);

private:
  cv::Ptr<cv::ORB> orb_;
};
