#pragma once
#include "frontend/feature_detector.hpp"
#include "frontend/feature_matcher.hpp"

struct Pose {
  cv::Mat R;   // 3x3 rotation matrix
  cv::Mat t;   // 3x1 translation vector
  int inliers; // number of matches RANSAC kept
};

class PoseEstimator {
public:
  virtual ~PoseEstimator() = default;

  virtual Pose estimate(const Features& features_a, const Features& features_b,
                        const std::vector<Match>& matches) = 0;
};
