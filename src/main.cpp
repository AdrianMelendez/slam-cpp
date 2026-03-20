#include <iostream>
#include "io/tum_loader.hpp"
#include "utils/logger.hpp"

int main(int argc, char** argv) {
    init_logger();

    if (argc < 2) {
        spdlog::error("Usage: slam_vo <path/to/tum/sequence>");
        return 1;
    }

    TUMLoader loader(argv[1]);
    if (!loader.load()) return 1;

    spdlog::info("Dataset ready: {} frames", loader.size());

    Frame f = loader.next();
    spdlog::info("First frame — id: {}  ts: {:.6f}  size: {}x{}",
                 f.id, f.timestamp, f.image.cols, f.image.rows);

    return 0;
}
