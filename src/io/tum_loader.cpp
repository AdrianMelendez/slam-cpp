#include "io/tum_loader.hpp"
#include <fstream>
#include <opencv2/imgcodecs.hpp> // cv::imread, cv::IMREAD_COLOR
#include <opencv2/imgproc.hpp>   // cv::cvtColor, cv::COLOR_BGR2GRAY
#include <spdlog/spdlog.h>

TUMLoader::TUMLoader(const std::string& sequence_path)
    : sequence_path_(sequence_path) {}

bool TUMLoader::load() {
  spdlog::debug("path: {}", sequence_path_ + "/rgb.txt");
  std::ifstream f(sequence_path_ + "/rgb.txt");
  if (!f.is_open()) {
    spdlog::error("Cannot open rgb.txt at {}", sequence_path_);
    return false;
  }

  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ss(line);
    double ts;
    std::string filename;
    ss >> ts >> filename;
    entries_.emplace_back(ts, filename);
  }

  spdlog::info("Loaded {} frames from {}", entries_.size(), sequence_path_);
  return !entries_.empty();
}

bool TUMLoader::has_next() const { return current_idx_ < entries_.size(); }

Frame TUMLoader::next() {
  const auto& [ts, rel_path] = entries_[current_idx_++];
  std::string full_path = sequence_path_ + "/" + rel_path;

  cv::Mat color = cv::imread(full_path, cv::IMREAD_COLOR);
  if (color.empty()) {
    spdlog::warn("Failed to read image: {}", full_path);
    return Frame{ts, cv::Mat(), frame_counter_++};
  }

  cv::Mat gray;
  cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

  spdlog::debug("Frame {} - ts: {:.6f} size {}x{}", frame_counter_, ts,
                gray.cols, gray.rows);

  return Frame{ts, gray, frame_counter_++};
}

size_t TUMLoader::size() const { return entries_.size(); }

std::vector<GroundTruthPose> TUMLoader::load_groundtruth() {
  std::vector<GroundTruthPose> poses;
  std::ifstream f(sequence_path_ + "/groundtruth.txt");

  if (!f.is_open()) {
    spdlog::warn("No groundtruth.txt found at {}", sequence_path_);
    return poses;
  }

  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ss(line);
    GroundTruthPose p;
    ss >> p.timestamp >> p.tx >> p.ty >> p.tz >> p.qx >> p.qy >> p.qz >> p.qw;
    poses.push_back(p);
  }

  spdlog::info("Loaded {} ground truth poses", poses.size());
  return poses;
}

// Build a 3x3 rotation matrix from a unit quaternion (TUM convention:
// qx,qy,qz,qw). Returns R such that p_world = R * p_local for the pose
// orientation.
cv::Mat quat_to_rot(double qx, double qy, double qz, double qw) {
  // normalize defensively — TUM GT quaternions are unit but floats drift
  double n = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (n > 0.0) {
    qx /= n;
    qy /= n;
    qz /= n;
    qw /= n;
  }

  cv::Mat R = (cv::Mat_<double>(3, 3) << 1 - 2 * (qy * qy + qz * qz),
               2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
               2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),
               2 * (qy * qz - qx * qw), 2 * (qx * qz - qy * qw),
               2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy));
  return R;
}

// Convert a 3x3 rotation matrix to a quaternion (TUM convention: qx,qy,qz,qw).
// Uses Shepperd's method for numerical stability.
void rot_to_quat(const cv::Mat& R, double& qx, double& qy, double& qz,
                 double& qw) {
  double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);

  if (trace > 0.0) {
    double s = 0.5 / std::sqrt(trace + 1.0);
    qw = 0.25 / s;
    qx = (R.at<double>(2, 1) - R.at<double>(1, 2)) * s;
    qy = (R.at<double>(0, 2) - R.at<double>(2, 0)) * s;
    qz = (R.at<double>(1, 0) - R.at<double>(0, 1)) * s;
  } else if (R.at<double>(0, 0) > R.at<double>(1, 1) &&
             R.at<double>(0, 0) > R.at<double>(2, 2)) {
    double s = 2.0 * std::sqrt(1.0 + R.at<double>(0, 0) - R.at<double>(1, 1) -
                               R.at<double>(2, 2));
    qw = (R.at<double>(2, 1) - R.at<double>(1, 2)) / s;
    qx = 0.25 * s;
    qy = (R.at<double>(0, 1) + R.at<double>(1, 0)) / s;
    qz = (R.at<double>(0, 2) + R.at<double>(2, 0)) / s;
  } else if (R.at<double>(1, 1) > R.at<double>(2, 2)) {
    double s = 2.0 * std::sqrt(1.0 + R.at<double>(1, 1) - R.at<double>(0, 0) -
                               R.at<double>(2, 2));
    qw = (R.at<double>(0, 2) - R.at<double>(2, 0)) / s;
    qx = (R.at<double>(0, 1) + R.at<double>(1, 0)) / s;
    qy = 0.25 * s;
    qz = (R.at<double>(1, 2) + R.at<double>(2, 1)) / s;
  } else {
    double s = 2.0 * std::sqrt(1.0 + R.at<double>(2, 2) - R.at<double>(0, 0) -
                               R.at<double>(1, 1));
    qw = (R.at<double>(1, 0) - R.at<double>(0, 1)) / s;
    qx = (R.at<double>(0, 2) + R.at<double>(2, 0)) / s;
    qy = (R.at<double>(1, 2) + R.at<double>(2, 1)) / s;
    qz = 0.25 * s;
  }
}

// Advance gt_idx to the GT sample closest to `t` (by absolute timestamp diff).
// Assumes ground truth is sorted by timestamp.
size_t nearest_gt(const std::vector<GroundTruthPose>& gt, size_t gt_idx,
                  double t) {
  while (gt_idx + 1 < gt.size() && gt[gt_idx + 1].timestamp <= t)
    ++gt_idx;
  if (gt_idx + 1 < gt.size()) {
    double d0 = std::abs(gt[gt_idx].timestamp - t);
    double d1 = std::abs(gt[gt_idx + 1].timestamp - t);
    if (d1 < d0)
      ++gt_idx;
  }
  return gt_idx;
}
