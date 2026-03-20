#pragma once
#include "frontend/feature_detector.hpp"
#include "opencv2/features2d.hpp"
#include <vector>

struct Match {
    int query_idx; // keypoint index in frame N
    int train_idx; // keypoint index in frame N+1
    float distance; // descriptor distance, lower = better
};

class FeatureMatcher {
public:
    explicit FeatureMatcher(float ratio_threshold = 0.7f);
    std::vector<Match> match(const Features& features_a, const Features& features_b);
private:
    cv::Ptr<cv::BFMatcher> matcher_;
    float ratio_threshold_;
};
