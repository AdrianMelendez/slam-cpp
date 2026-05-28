#pragma once
#include "frontend/pose_estimator.hpp"
#include "io/tum_loader.hpp"
#include "opencv2/core.hpp"

// RGB-D pose estimation via PnP.
// Backprojects keypoints in frame A to 3D using the depth channel, then
// solves the 3D-2D correspondence problem against frame B. Unlike the
// monocular estimator, translation is metric (no scale ambiguity).
class RGBDPoseEstimator : public PoseEstimator {
public:
  explicit RGBDPoseEstimator(const TUMLoader::Intrinsics& intrinsics);

  Pose estimate(const Features& features_a, const Features& features_b,
                const std::vector<Match>& matches) override;

private:
  TUMLoader::Intrinsics intrinsics_;
  cv::Mat K_;
  cv::Mat dist_coeffs_; // TUM images are pre-undistorted → zeros
};
