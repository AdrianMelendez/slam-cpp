#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct Frame {
  double timestamp;
  cv::Mat image;  // grayscale uint8
  cv::Mat depth;  // float32 meters, empty if unavailable
  int id;
};

struct GroundTruthPose {
  double timestamp;
  double tx, ty, tz;     // position
  double qx, qy, qz, qw; // orientation (quaternion)
};

cv::Mat quat_to_rot(double qx, double qy, double qz, double qw);
size_t nearest_gt(const std::vector<GroundTruthPose>& gt, size_t gt_idx,
                  double t);
void rot_to_quat(const cv::Mat& R, double& qx, double& qy, double& qz,
                 double& qw);

class TUMLoader {
public:
  explicit TUMLoader(const std::string& sequence_path);

  bool load();
  bool has_next() const;
  Frame next();
  size_t size() const;

  struct Intrinsics {
    double fx = 517.306408;
    double fy = 516.469215;
    double cx = 318.643040;
    double cy = 255.313989;
    double depth_scale = 5000.0; // raw pixel / depth_scale = meters (TUM default)
  } intrinsics;

  std::vector<GroundTruthPose> load_groundtruth();

private:
  std::string sequence_path_;
  std::vector<std::pair<double, std::string>> entries_;
  std::vector<std::string> matched_depth_paths_; // per-entry depth path, empty if unavailable
  size_t current_idx_ = 0;
  int frame_counter_ = 0;
};
