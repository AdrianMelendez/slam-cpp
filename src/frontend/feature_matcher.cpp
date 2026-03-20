#include "frontend/feature_matcher.hpp"
#include "frontend/feature_detector.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include <spdlog/spdlog.h>
#include <vector>

FeatureMatcher::FeatureMatcher(float ratio_threshold) : ratio_threshold_(ratio_threshold){
    // NORM_HAMMING is the correct distance metric for ORB descriptors
    // ORB descriptors are binary strings, Hamming distance counts differing bits
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
    spdlog::debug("FeatureMatcher initialized with ratio threshold {:.2f}",ratio_threshold_);
}

std::vector<Match> FeatureMatcher::match(const Features& features_a, const Features& features_b) {
    // knnMatch returns the 2 best matches for each descriptor
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(features_a.descriptors, features_b.descriptors, knn_matches, 2);

    // Lowe's ratio test - discard ambiguous matches
    std::vector<Match> good_matches;
    for (const auto& pair : knn_matches){
        if (pair.size() < 2) continue;
        if (pair[0].distance < ratio_threshold_ * pair[1].distance) {
            good_matches.push_back({
                pair[0].queryIdx,
                pair[0].trainIdx,
                pair[0].distance
            });
        }
    }

    spdlog::debug("Matched {} / {} raw pair passed ratio test", good_matches.size(), knn_matches.size());
    return good_matches;
}
