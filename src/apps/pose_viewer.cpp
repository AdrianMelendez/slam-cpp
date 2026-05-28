#include "frontend/feature_detector.hpp"
#include "frontend/feature_matcher.hpp"
#include "frontend/pose_estimator.hpp"
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
constexpr int kScaleInitFrames = 10; // frames used to calibrate scale
constexpr double kMinStep = 1e-4;    // ignore tiny GT steps when calibrating
constexpr float kPixelsPerMeter = 100.0f; // canvas zoom

} // namespace

int main(int argc, char** argv) {
  init_logger();

  if (argc < 2) {
    spdlog::error("Usage: pose_viewer <path/to/tum/sequence> [--output "
                  "<trajectory.txt>]");
    return 1;
  }

  std::string output_path;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    }
  }
  if (!output_path.empty()) {
    spdlog::info("Trajectory will be saved to: {}", output_path);
  }

  TUMLoader loader(argv[1]);
  if (!loader.load())
    return 1;

  auto groundtruth = loader.load_groundtruth();
  const bool has_gt = !groundtruth.empty();

  FeatureDetector detector;
  FeatureMatcher matcher;
  PoseEstimator estimator(loader.intrinsics);

  Frame prev_frame = loader.next();
  Features prev_features = detector.detect(prev_frame);

  // accumulated camera pose in the world frame (start at origin)
  cv::Mat R_total = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat t_total = cv::Mat::zeros(3, 1, CV_64F);

  // Align ground truth to the first camera's coordinate frame:
  //   GT is reported as T_w_c (camera-in-world, Vicon frame). We want each
  //   GT point expressed in the *first camera's* frame so it matches our
  //   estimated trajectory (which starts at identity in camera-0 frame):
  //       p_c0 = R_w_c0^T * (p_w - p_w_c0)
  size_t gt_idx = 0;
  cv::Mat t_w_c0 = cv::Mat::zeros(3, 1, CV_64F); // GT origin in world frame
  cv::Mat R_c0_w = cv::Mat::eye(3, 3, CV_64F);   // world -> camera-0 rotation
  if (has_gt) {
    gt_idx = nearest_gt(groundtruth, 0, prev_frame.timestamp);
    const auto& g0 = groundtruth[gt_idx];
    t_w_c0.at<double>(0) = g0.tx;
    t_w_c0.at<double>(1) = g0.ty;
    t_w_c0.at<double>(2) = g0.tz;
    cv::Mat R_w_c0 = quat_to_rot(g0.qx, g0.qy, g0.qz, g0.qw);
    R_c0_w = R_w_c0.t();
  } else {
    spdlog::warn("No ground truth available - overlay disabled");
  }

  // monocular scale is unobservable from epipolar geometry alone.
  // We calibrate it once: ratio of ||GT step|| to ||estimated step||,
  // averaged over the first kScaleInitFrames pairs, then frozen.
  double scale = 1.0;
  bool scale_locked = !has_gt; // if no GT, just plot in arbitrary units
  double scale_num_sum = 0.0;  // sum of GT step magnitudes
  double scale_den_sum = 0.0;  // sum of estimated step magnitudes
  int scale_samples = 0;

  // canvas for the top-down trajectory plot
  cv::Mat canvas(kCanvasSize, kCanvasSize, CV_8UC3, cv::Scalar(30, 30, 30));
  const cv::Point2i center(kCanvasSize / 2, kCanvasSize / 2);

  // previous positions for step-magnitude calculation
  cv::Vec2d prev_est_xz(0.0, 0.0);
  cv::Vec2d prev_gt_xz(0.0, 0.0);

  // collect poses for output: (timestamp, tx, ty, tz, qx, qy, qz, qw)
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

    // skip frames where RANSAC failed - keep prev_* anchored to last good frame
    // so the next iteration matches against it instead of compounding the gap
    if (pose.inliers == 0) {
      spdlog::warn("Frame {} - pose estimation failed, skipping",
                   curr_frame.id);
      continue;
    }

    // pose.R, pose.t express points in curr-cam given points in prev-cam:
    //   X_curr = R * X_prev + t
    // To accumulate T_wc (camera-in-world), compose with the *inverse*:
    //   T_wc_curr = T_wc_prev * inv(R, t)
    cv::Mat R_rel = pose.R.t();
    cv::Mat t_rel = -pose.R.t() * pose.t;

    t_total = t_total + R_total * t_rel;
    R_total = R_total * R_rel;

    // collect pose for output
    double qx, qy, qz, qw;
    rot_to_quat(R_total, qx, qy, qz, qw);
    estimated_poses.push_back({curr_frame.timestamp, t_total.at<double>(0),
                               t_total.at<double>(1), t_total.at<double>(2), qx,
                               qy, qz, qw});

    // ---- ground-truth alignment in camera-0 frame ----
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

    // ---- monocular scale calibration ----
    cv::Vec2d est_xz(t_total.at<double>(0), t_total.at<double>(2));
    if (!scale_locked && gt_valid) {
      double est_step = cv::norm(est_xz - prev_est_xz);
      double gt_step = cv::norm(gt_xz - prev_gt_xz);
      if (est_step > kMinStep && gt_step > kMinStep) {
        scale_num_sum += gt_step;
        scale_den_sum += est_step;
        ++scale_samples;
      }
      if (scale_samples >= kScaleInitFrames) {
        scale = scale_num_sum / scale_den_sum;
        scale_locked = true;
        spdlog::info("Scale calibrated: {:.4f} (over {} samples)", scale,
                     scale_samples);
      }
    }
    prev_est_xz = est_xz;
    prev_gt_xz = gt_xz;

    // ---- draw on canvas ----
    // skip estimated dots until scale is locked, otherwise the warmup samples
    // (plotted with scale=1.0) streak off the canvas as outliers
    if (scale_locked) {
      int x = center.x + static_cast<int>(est_xz[0] * scale * kPixelsPerMeter);
      int z = center.y - static_cast<int>(est_xz[1] * scale * kPixelsPerMeter);
      if (x >= 0 && x < kCanvasSize && z >= 0 && z < kCanvasSize)
        cv::circle(canvas, cv::Point(x, z), 2, cv::Scalar(0, 255, 0), -1);
    }

    if (gt_valid) {
      int gx = center.x + static_cast<int>(gt_xz[0] * kPixelsPerMeter);
      int gz = center.y - static_cast<int>(gt_xz[1] * kPixelsPerMeter);
      if (gx >= 0 && gx < kCanvasSize && gz >= 0 && gz < kCanvasSize)
        cv::circle(canvas, cv::Point(gx, gz), 2, cv::Scalar(0, 0, 255), -1);
    }

    // clone canvas before stamping the legend so trajectory dots don't
    // overwrite it across frames
    cv::Mat canvas_display = canvas.clone();
    cv::circle(canvas_display, cv::Point(20, 20), 4, cv::Scalar(0, 255, 0), -1);
    cv::putText(canvas_display, "estimated", cv::Point(30, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    cv::circle(canvas_display, cv::Point(20, 40), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(canvas_display, "ground truth", cv::Point(30, 45),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
    if (!scale_locked) {
      cv::putText(canvas_display, "calibrating scale...", cv::Point(20, 65),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
    }

    // ---- frame overlay ----
    cv::Mat display;
    cv::cvtColor(curr_frame.image, display, cv::COLOR_GRAY2BGR);
    cv::putText(display,
                "matches: " + std::to_string(matches.size()) +
                    "  inliers: " + std::to_string(pose.inliers),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 1);

    cv::imshow("Frame", display);
    cv::imshow("Trajectory", canvas_display);

    if ((cv::waitKey(33) & 0xFF) == 'q')
      break;

    prev_features = curr_features;
  }

  cv::destroyAllWindows();

  // save trajectory if output path was specified
  if (!output_path.empty() && !estimated_poses.empty()) {
    std::ofstream out(output_path);
    if (out.is_open()) {
      // write each pose in TUM format: timestamp tx ty tz qx qy qz qw
      out << std::fixed << std::setprecision(6);
      for (const auto& pose : estimated_poses) {
        out << pose.timestamp << " " << pose.tx << " " << pose.ty << " "
            << pose.tz << " " << pose.qx << " " << pose.qy << " " << pose.qz
            << " " << pose.qw << "\n";
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
