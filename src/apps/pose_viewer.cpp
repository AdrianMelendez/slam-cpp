#include "frontend/feature_detector.hpp"
#include "frontend/feature_matcher.hpp"
#include "frontend/pose_estimator.hpp"
#include "io/tum_loader.hpp"
#include "utils/logger.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
  init_logger();

  if (argc < 2) {
    spdlog::error("Usage: pose_viewer <path/to/tum/sequence>");
    return 1;
  }

  TUMLoader loader(argv[1]);
  if (!loader.load())
    return 1;
  auto groundtruth = loader.load_groundtruth();
  size_t gt_idx = 0;

  FeatureDetector detector;
  FeatureMatcher matcher;
  PoseEstimator estimator(loader.intrinsics);

  Frame prev_frame = loader.next();
  Features prev_features = detector.detect(prev_frame);

  // accumulate trajectory — start at origin
  cv::Mat R_total = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat t_total = cv::Mat::zeros(3, 1, CV_64F);

  // canvas to draw the top-down trajectory
  int canvas_size = 600;
  cv::Mat canvas(canvas_size, canvas_size, CV_8UC3, cv::Scalar(30, 30, 30));
  cv::Point2i center(canvas_size / 2, canvas_size / 2);
  // offset
  center.x += 1.3563;
  center.y += 1.6380;

  // draw legend once before the loop
  cv::circle(canvas, cv::Point(20, 20), 4, cv::Scalar(0, 255, 0), -1);
  cv::putText(canvas, "estimated", cv::Point(30, 25), cv::FONT_HERSHEY_SIMPLEX,
              0.4, cv::Scalar(0, 255, 0), 1);
  cv::circle(canvas, cv::Point(20, 40), 4, cv::Scalar(0, 0, 255), -1);
  cv::putText(canvas, "ground truth", cv::Point(30, 45),
              cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

  while (loader.has_next()) {
    Frame curr_frame = loader.next();
    Features curr_features = detector.detect(curr_frame);
    std::vector<Match> matches = matcher.match(prev_features, curr_features);
    Pose pose = estimator.estimate(prev_features, curr_features, matches);

    // pose.R, pose.t : transform points from prev-cam into curr-cam (X_curr = R
    // X_prev + t) We want T_wc_curr = T_wc_prev * T_prev_curr, where
    // T_prev_curr is the camera's motion expressed in prev-cam frame = inverse
    // of (R, t):
    cv::Mat R_rel = pose.R.t();
    cv::Mat t_rel = -pose.R.t() * pose.t;

    t_total = t_total + R_total * t_rel;
    R_total = R_total * R_rel;

    // project the 3D position onto a top-down 2D canvas (X-Z plane)
    // scale factor controls how many pixels per meter
    float scale = 10.0f;
    int x = center.x + static_cast<int>(t_total.at<double>(0) * scale);
    int z = center.y + static_cast<int>(t_total.at<double>(2) * scale);

    // draw estimated pose in green
    cv::circle(canvas, cv::Point(x, z), 2, cv::Scalar(0, 255, 0), -1);

    // draw groundtruth in red
    while (gt_idx + 1 < groundtruth.size() &&
           groundtruth[gt_idx].timestamp < curr_frame.timestamp)
      gt_idx++;

    // use a much larger scale for ground truth to make it visible
    float gt_scale = scale;
    if (gt_idx < groundtruth.size()) {
      const auto& gt = groundtruth[gt_idx];
      int gt_x = center.x + static_cast<int>(gt.tx * gt_scale);
      int gt_z = center.y + static_cast<int>(gt.tz * gt_scale);
      cv::circle(canvas, cv::Point(gt_x, gt_z), 2, cv::Scalar(0, 0, 255),
                 -1); // red = ground truth
    }

    // show match count and inliers on the frame
    cv::Mat display = curr_frame.image.clone();
    cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);
    cv::putText(display,
                "matches: " + std::to_string(matches.size()) +
                    "  inliers: " + std::to_string(pose.inliers),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 1);

    cv::imshow("Frame", display);
    cv::imshow("Trajectory", canvas);

    if ((cv::waitKey(33) & 0xFF) == 'q')
      break;

    prev_frame = curr_frame;
    prev_features = curr_features;
  }

  cv::destroyAllWindows();
  return 0;
}
