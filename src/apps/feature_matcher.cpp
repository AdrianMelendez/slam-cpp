#include "frontend/feature_matcher.hpp"
#include "frontend/feature_detector.hpp"
#include "io/tum_loader.hpp"
#include "utils/logger.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
  init_logger();

  if (argc < 2) {
    spdlog::error("Usage: feature_matcher <path/to/tum/sequence>");
    return 1;
  }

  TUMLoader loader(argv[1]);
  if (!loader.load())
    return 1;

  spdlog::info("Dataset ready: {} frames", loader.size());

  FeatureDetector detector;
  FeatureMatcher matcher;

  Frame prev_frame = loader.next();
  Features prev_features = detector.detect(prev_frame);

  while (loader.has_next()) {
    Frame curr_frame = loader.next();
    Features curr_features = detector.detect(curr_frame);

    std::vector<Match> matches = matcher.match(prev_features, curr_features);

    // convert our Match structs back to cv::DMatch for drawing
    std::vector<cv::DMatch> cv_matches;
    for (const auto& m : matches)
      cv_matches.push_back(cv::DMatch(m.query_idx, m.train_idx, m.distance));

    cv::Mat display;
    cv::drawMatches(prev_frame.image, prev_features.keypoints, curr_frame.image,
                    curr_features.keypoints, cv_matches, display,
                    cv::Scalar(0, 255, 0), // match color
                    cv::Scalar(255, 0, 0), // single point color
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Matches", display);
    if ((cv::waitKey(33) & 0xFF) == 'q')
      break;

    prev_frame = curr_frame;
    prev_features = curr_features;
  }

  cv::destroyAllWindows();
  return 0;
}
