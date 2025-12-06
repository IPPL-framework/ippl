#pragma once

#if defined(IPPL_LOGGING_ENABLED)

#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <tuple>
#include "Utility/PrintType.h"

template <typename... Args>
struct scoped_var {
    // capture tuple elements by reference - no temp vars in constructor please
    std::tuple<Args const&...> const message_;
    //
    explicit scoped_var(Args const&... args)
        : message_(args...)  //
    {
        SPDLOG_CRITICAL("SCOPE >> enter << {}", message_);
    }

    ~scoped_var() { SPDLOG_CRITICAL("SCOPE << leave >> {}", message_); }
};
#define SPDLOG_SCOPE(...) scoped_var scope(__VA_ARGS__);

#else

// In increasing level
#define SPDLOG_TRACE(...)
#define SPDLOG_DEBUG(...)
#define SPDLOG_INFO(...)
#define SPDLOG_WARN(...)
#define SPDLOG_ERROR(...)
#define SPDLOG_CRITICAL(...)
//
#define SPDLOG_SCOPE(...)

#endif
