#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

inline void init_logger(spdlog::level::level_enum level = spdlog::level::debug) {
    auto logger = spdlog::stdout_color_mt("slam");
    logger->set_level(level);
    logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] [%n] %v");
    spdlog::set_default_logger(logger);
}
