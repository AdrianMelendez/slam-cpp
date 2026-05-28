#include "frontend/feature_detector.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/features2d.hpp"
#include <spdlog/spdlog.h>

FeatureDetector::FeatureDetector(int n_features) {
  orb_ = cv::ORB::create(n_features);
  spdlog::debug("FeatureDetector initialized with {} max features", n_features);
}

Features FeatureDetector::detect(const Frame& frame) {
  Features features;
  orb_->detectAndCompute(frame.image, cv::noArray(), features.keypoints,
                         features.descriptors);
  features.depth_map = frame.depth;
  spdlog::debug("Frame {} - detected {} keypoints", frame.id,
                features.keypoints.size());
  return features;
}
