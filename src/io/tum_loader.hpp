#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct Frame {
  double timestamp;
  cv::Mat image;
  int id;
};

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
  } intrinsics;

private:
  std::string sequence_path_;
  std::vector<std::pair<double, std::string>> entries_;
  size_t current_idx_ = 0;
  int frame_counter_ = 0;
};
