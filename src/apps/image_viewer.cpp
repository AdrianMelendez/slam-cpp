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
        spdlog::error("Usage: image_viewer <path/to/tum/sequence>");
        return 1;
    }

    TUMLoader loader(argv[1]);
    if (!loader.load()) return 1;

    spdlog::info("Dataset ready: {} frames", loader.size());

    while (loader.has_next()) {
            Frame f = loader.next();
            cv::imshow("TUM Sequence", f.image);
            // wait 33ms between frames (~30fps), quit if 'q' is pressed
            if (cv::waitKey(50) == 'q') break;
        }

    cv::destroyAllWindows();
    return 0;
}
