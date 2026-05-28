#pragma once
#include "frontend/pose_estimator.hpp"
#include "io/tum_loader.hpp"
#include "opencv2/core.hpp"

// Monocular pose estimation via the essential matrix.
// Recovers relative pose up to scale (translation is a unit vector).
class MonocularPoseEstimator : public PoseEstimator {
public:
  explicit MonocularPoseEstimator(const TUMLoader::Intrinsics& intrinsics);

  Pose estimate(const Features& features_a, const Features& features_b,
                const std::vector<Match>& matches) override;

private:
  TUMLoader::Intrinsics intrinsics_;
  cv::Mat K_; // 3x3 camera matrix

  // pull matched pixel coordinates out of a keypoint set
  std::vector<cv::Point2f> to_points(const std::vector<cv::KeyPoint>& keypoints,
                                     const std::vector<Match>& matches,
                                     bool query) const;
};
