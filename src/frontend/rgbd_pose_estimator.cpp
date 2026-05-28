#include "frontend/rgbd_pose_estimator.hpp"
#include "spdlog/spdlog.h"
#include <cmath>
#include <opencv2/calib3d.hpp>

RGBDPoseEstimator::RGBDPoseEstimator(const TUMLoader::Intrinsics& intrinsics)
    : intrinsics_(intrinsics) {
  K_ = (cv::Mat_<double>(3, 3) << intrinsics_.fx, 0, intrinsics_.cx, 0,
        intrinsics_.fy, intrinsics_.cy, 0, 0, 1.0);
  dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64F);

  spdlog::debug(
      "RGBDPoseEstimator initialized - fx:{:.2f} fy:{:.2f} cx:{:.2f} cy:{:.2f}",
      intrinsics_.fx, intrinsics_.fy, intrinsics_.cx, intrinsics_.cy);
}

Pose RGBDPoseEstimator::estimate(const Features& features_a,
                                  const Features& features_b,
                                  const std::vector<Match>& matches) {
  Pose pose;
  pose.R = cv::Mat::eye(3, 3, CV_64F);
  pose.t = cv::Mat::zeros(3, 1, CV_64F);
  pose.inliers = 0;

  if (features_a.depth_map.empty()) {
    spdlog::warn("No depth map in frame A — cannot estimate RGB-D pose");
    return pose;
  }

  const double fx = intrinsics_.fx;
  const double fy = intrinsics_.fy;
  const double cx = intrinsics_.cx;
  const double cy = intrinsics_.cy;

  // For each match, backproject the keypoint in A to 3D using its depth value,
  // and record the corresponding 2D keypoint in B.
  std::vector<cv::Point3d> points3d;
  std::vector<cv::Point2d> points2d;
  points3d.reserve(matches.size());
  points2d.reserve(matches.size());

  for (const auto& m : matches) {
    const cv::Point2f& pt_a = features_a.keypoints[m.query_idx].pt;
    int u = static_cast<int>(std::round(pt_a.x));
    int v = static_cast<int>(std::round(pt_a.y));

    if (u < 0 || v < 0 || u >= features_a.depth_map.cols ||
        v >= features_a.depth_map.rows)
      continue;

    double depth = static_cast<double>(features_a.depth_map.at<float>(v, u));
    if (depth <= 0.0 || !std::isfinite(depth))
      continue;

    points3d.push_back({(pt_a.x - cx) * depth / fx,
                        (pt_a.y - cy) * depth / fy,
                        depth});
    points2d.push_back({features_b.keypoints[m.train_idx].pt.x,
                        features_b.keypoints[m.train_idx].pt.y});
  }

  if (static_cast<int>(points3d.size()) < 6) {
    spdlog::warn("Not enough 3D-2D correspondences for PnP: {}",
                 points3d.size());
    return pose;
  }

  cv::Mat rvec, tvec;
  std::vector<int> inliers;
  bool ok = cv::solvePnPRansac(points3d, points2d, K_, dist_coeffs_, rvec, tvec,
                                false,  // useExtrinsicGuess
                                100,    // iterations
                                2.0,    // reprojection error threshold (px)
                                0.999,  // confidence
                                inliers);

  if (!ok || inliers.empty()) {
    spdlog::warn("solvePnPRansac failed");
    return pose;
  }

  cv::Rodrigues(rvec, pose.R);
  pose.t = tvec;
  pose.inliers = static_cast<int>(inliers.size());

  spdlog::debug("RGB-D pose estimated - inliers: {}/{}", pose.inliers,
                points3d.size());
  return pose;
}
