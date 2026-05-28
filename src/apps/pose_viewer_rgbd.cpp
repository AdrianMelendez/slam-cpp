#include "frontend/feature_detector.hpp"
#include "frontend/feature_matcher.hpp"
#include "frontend/rgbd_pose_estimator.hpp"
#include "io/tum_loader.hpp"
#include "utils/logger.hpp"
#include <fstream>
#include <iomanip>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace {

constexpr int kCanvasSize = 600;
constexpr float kPixelsPerMeter = 100.0f;

} // namespace

int main(int argc, char** argv) {
  init_logger();

  if (argc < 2) {
    spdlog::error("Usage: pose_viewer_rgbd <path/to/tum/sequence> [--output "
                  "<trajectory.txt>]");
    return 1;
  }

  std::string output_path;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--output" && i + 1 < argc)
      output_path = argv[++i];
  }
  if (!output_path.empty())
    spdlog::info("Trajectory will be saved to: {}", output_path);

  TUMLoader loader(argv[1]);
  if (!loader.load())
    return 1;

  auto groundtruth = loader.load_groundtruth();
  const bool has_gt = !groundtruth.empty();

  FeatureDetector detector;
  FeatureMatcher matcher;
  RGBDPoseEstimator estimator(loader.intrinsics);

  Frame prev_frame = loader.next();
  if (prev_frame.depth.empty())
    spdlog::warn("First frame has no depth — RGB-D estimation will degrade to "
                 "zero pose");

  Features prev_features = detector.detect(prev_frame);

  cv::Mat R_total = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat t_total = cv::Mat::zeros(3, 1, CV_64F);

  size_t gt_idx = 0;
  cv::Mat t_w_c0 = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat R_c0_w = cv::Mat::eye(3, 3, CV_64F);
  if (has_gt) {
    gt_idx = nearest_gt(groundtruth, 0, prev_frame.timestamp);
    const auto& g0 = groundtruth[gt_idx];
    t_w_c0.at<double>(0) = g0.tx;
    t_w_c0.at<double>(1) = g0.ty;
    t_w_c0.at<double>(2) = g0.tz;
    cv::Mat R_w_c0 = quat_to_rot(g0.qx, g0.qy, g0.qz, g0.qw);
    R_c0_w = R_w_c0.t();
  } else {
    spdlog::warn("No ground truth available — overlay disabled");
  }

  cv::Mat canvas(kCanvasSize, kCanvasSize, CV_8UC3, cv::Scalar(30, 30, 30));
  const cv::Point2i center(kCanvasSize / 2, kCanvasSize / 2);

  struct EstimatedPose {
    double timestamp;
    double tx, ty, tz;
    double qx, qy, qz, qw;
  };
  std::vector<EstimatedPose> estimated_poses;

  while (loader.has_next()) {
    Frame curr_frame = loader.next();
    Features curr_features = detector.detect(curr_frame);

    std::vector<Match> matches = matcher.match(prev_features, curr_features);
    Pose pose = estimator.estimate(prev_features, curr_features, matches);

    if (pose.inliers == 0) {
      spdlog::warn("Frame {} — pose estimation failed, skipping", curr_frame.id);
      continue;
    }

    // pose.(R,t) maps points from prev frame to curr frame: X_curr = R*X_prev + t
    // Invert to get the camera's motion through world
    cv::Mat R_rel = pose.R.t();
    cv::Mat t_rel = -pose.R.t() * pose.t;

    t_total = t_total + R_total * t_rel;
    R_total = R_total * R_rel;

    double qx, qy, qz, qw;
    rot_to_quat(R_total, qx, qy, qz, qw);
    estimated_poses.push_back({curr_frame.timestamp, t_total.at<double>(0),
                               t_total.at<double>(1), t_total.at<double>(2), qx,
                               qy, qz, qw});

    // ---- ground-truth in camera-0 frame ----
    cv::Vec2d gt_xz(0.0, 0.0);
    bool gt_valid = false;
    if (has_gt) {
      gt_idx = nearest_gt(groundtruth, gt_idx, curr_frame.timestamp);
      const auto& g = groundtruth[gt_idx];
      cv::Mat p_w = (cv::Mat_<double>(3, 1) << g.tx - t_w_c0.at<double>(0),
                     g.ty - t_w_c0.at<double>(1), g.tz - t_w_c0.at<double>(2));
      cv::Mat p_c0 = R_c0_w * p_w;
      gt_xz = {p_c0.at<double>(0), p_c0.at<double>(2)};
      gt_valid = true;
    }

    // ---- draw trajectory ----
    cv::Vec2d est_xz(t_total.at<double>(0), t_total.at<double>(2));

    int x = center.x + static_cast<int>(est_xz[0] * kPixelsPerMeter);
    int z = center.y - static_cast<int>(est_xz[1] * kPixelsPerMeter);
    if (x >= 0 && x < kCanvasSize && z >= 0 && z < kCanvasSize)
      cv::circle(canvas, cv::Point(x, z), 2, cv::Scalar(0, 255, 0), -1);

    if (gt_valid) {
      int gx = center.x + static_cast<int>(gt_xz[0] * kPixelsPerMeter);
      int gz = center.y - static_cast<int>(gt_xz[1] * kPixelsPerMeter);
      if (gx >= 0 && gx < kCanvasSize && gz >= 0 && gz < kCanvasSize)
        cv::circle(canvas, cv::Point(gx, gz), 2, cv::Scalar(0, 0, 255), -1);
    }

    cv::Mat canvas_display = canvas.clone();
    cv::circle(canvas_display, cv::Point(20, 20), 4, cv::Scalar(0, 255, 0), -1);
    cv::putText(canvas_display, "estimated (RGB-D)", cv::Point(30, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    cv::circle(canvas_display, cv::Point(20, 40), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(canvas_display, "ground truth", cv::Point(30, 45),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

    // ---- RGB frame overlay ----
    cv::Mat display;
    cv::cvtColor(curr_frame.image, display, cv::COLOR_GRAY2BGR);
    cv::putText(display,
                "matches: " + std::to_string(matches.size()) +
                    "  inliers: " + std::to_string(pose.inliers),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 1);

    cv::imshow("Frame", display);
    cv::imshow("Trajectory", canvas_display);

    // ---- depth visualization ----
    if (!curr_frame.depth.empty()) {
      cv::Mat depth_vis;
      cv::Mat depth_normalized;
      // clamp to 0-5m range for a stable color scale
      cv::Mat depth_clamped;
      cv::min(curr_frame.depth, 5.0f, depth_clamped);
      cv::normalize(depth_clamped, depth_normalized, 0, 255, cv::NORM_MINMAX,
                    CV_8U);
      cv::applyColorMap(depth_normalized, depth_vis, cv::COLORMAP_JET);
      cv::imshow("Depth", depth_vis);
    }

    if ((cv::waitKey(33) & 0xFF) == 'q')
      break;

    prev_features = curr_features;
  }

  cv::destroyAllWindows();

  if (!output_path.empty() && !estimated_poses.empty()) {
    std::ofstream out(output_path);
    if (out.is_open()) {
      out << std::fixed << std::setprecision(6);
      for (const auto& p : estimated_poses) {
        out << p.timestamp << " " << p.tx << " " << p.ty << " " << p.tz << " "
            << p.qx << " " << p.qy << " " << p.qz << " " << p.qw << "\n";
      }
      out.close();
      spdlog::info("Trajectory saved to: {} ({} poses)", output_path,
                   estimated_poses.size());
    } else {
      spdlog::error("Failed to open output file: {}", output_path);
    }
  }

  return 0;
}
