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
