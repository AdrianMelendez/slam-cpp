#pragma once
#include "frontend/feature_detector.hpp"
#include "frontend/feature_matcher.hpp"
#include "io/tum_loader.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"

struct Pose {
  cv::Mat R;   // 3x3 rotation matrix
  cv::Mat t;   // 3x1 translation vector
  int inliers; // number of matches RANSAC kept
};

class PoseEstimator {
public:
  explicit PoseEstimator(const TUMLoader::Intrinsics& intrinsics);

  Pose estimate(const Features& features_a, const Features& features_b,
                const std::vector<Match>& matches);

private:
  TUMLoader::Intrinsics intrinsics_;
  cv::Mat K_; // 3x3 camera matrix

  // convert pixel keypoints to normalized image coordinates
  std::vector<cv::Point2f> to_points(const std::vector<cv::KeyPoint>& keypoints,
                                     const std::vector<Match>& matches,
                                     bool query) const;
};
