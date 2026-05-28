#include "frontend/monocular_pose_estimator.hpp"
#include "frontend/feature_detector.hpp"
#include "pose_estimator.hpp"
#include "spdlog/spdlog.h"
#include <opencv2/calib3d.hpp>

MonocularPoseEstimator::MonocularPoseEstimator(
    const TUMLoader::Intrinsics& intrinsics)
    : intrinsics_(intrinsics) {
  K_ = (cv::Mat_<double>(3, 3) << intrinsics_.fx, 0, intrinsics_.cx, 0,
        intrinsics_.fy, intrinsics_.cy, 0, 0, 1.0);

  spdlog::debug(
      "PoseEstimator initialized - fx:{:.2f} fy:{:.2f} cx:{:.2f} cy:{:.2f}",
      intrinsics_.fx, intrinsics_.fy, intrinsics_.cx, intrinsics_.cy);
};

std::vector<cv::Point2f>
MonocularPoseEstimator::to_points(const std::vector<cv::KeyPoint>& keypoints,
                                  const std::vector<Match>& matches,
                                  bool query) const {
  std::vector<cv::Point2f> pts;
  pts.reserve(matches.size());
  for (const auto& m : matches) {
    int idx = query ? m.query_idx : m.train_idx;
    pts.push_back(keypoints[idx].pt);
  }
  return pts;
}

Pose MonocularPoseEstimator::estimate(const Features& features_a,
                                      const Features& features_b,
                                      const std::vector<Match>& matches) {
  Pose pose;
  pose.R = cv::Mat::eye(3, 3, CV_64F);
  pose.t = cv::Mat::zeros(3, 1, CV_64F);
  pose.inliers = 0;

  if (matches.size() < 8) {
    spdlog::warn("Not enough matches for pose estimation: {}", matches.size());
    return pose;
  }

  // extract matched pixel coordinates from both frames
  std::vector<cv::Point2f> pts_a =
      to_points(features_a.keypoints, matches, true);
  std::vector<cv::Point2f> pts_b =
      to_points(features_b.keypoints, matches, false);

  // compute essential matrix with RANSAC
  // RANSAC threshold of 1.0 pixel — matches that don't fit the epipolar
  // constraint within 1px are treated as outliers
  std::vector<uchar> inlier_mask;
  cv::Mat E = cv::findEssentialMat(pts_a, pts_b, K_, cv::RANSAC, 0.999, 1.0,
                                   inlier_mask);

  if (E.empty()) {
    spdlog::warn("Essential matrix computation failed");
    return pose;
  }

  // decompose E into R and t, keeping only inliers
  // recoverPose also picks the correct solution out of the 4 possible
  // decompositions by checking that points are in front of both cameras
  pose.inliers =
      cv::recoverPose(E, pts_a, pts_b, K_, pose.R, pose.t, inlier_mask);

  spdlog::debug("Pose estimated - inliners: {}/{}", pose.inliers,
                matches.size());
  return pose;
}
