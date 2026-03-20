#include <iostream>
#include <opencv2/highgui.hpp>
#include "frontend/feature_detector.hpp"
#include "io/tum_loader.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/features2d.hpp"
#include "utils/logger.hpp"

int main(int argc, char** argv) {
    init_logger();

    if (argc < 2) {
        spdlog::error("Usage: slam_cpp <path/to/tum/sequence>");
        return 1;
    }

    TUMLoader loader(argv[1]);
    if (!loader.load()) return 1;

    spdlog::info("Dataset ready: {} frames", loader.size());

    FeatureDetector detector;

    while (loader.has_next()) {
            Frame f = loader.next();

            Features features = detector.detect(f);

            // draw keypoints on a copy of the frame
            cv::Mat display;
            cv::drawKeypoints(f.image, features.keypoints, display, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            cv::imshow("Features", display);

            // wait 33ms between frames (~30fps), quit if 'q' is pressed
            if (cv::waitKey(50) == 'q') break;
        }

    cv::destroyAllWindows();
    return 0;
}
