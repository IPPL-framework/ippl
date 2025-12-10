#pragma once

#if defined(IPPL_LOGGING_ENABLED)

#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <source_location>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <tuple>

#include "Utility/PrintType.h"

namespace ippl::debug {
    // a singleton logger for scope messages
    inline auto scope_logger = spdlog::stdout_color_mt("console");

    static int logger_init() {
        scope_logger->set_pattern("[scope %t| %v");
        return 0;
    }

    inline int log_init_{logger_init()};

    template <typename... Args>
    struct scoped_var {
        // capture tuple elements by reference - no temp vars in constructor please
        std::tuple<Args const&...> const message_;
        //
        template <class... Ts, std::size_t... Is, class Tuple>
        decltype(auto) tie_from_specified(std::index_sequence<Is...>, Tuple& tuple) {
            return std::tuple<Ts...>{std::get<Is>(tuple)...};
        }
        template <class... Ts, class Tuple>
        decltype(auto) tie_from(Tuple& tuple) {
            return tie_from_specified<Ts...>(std::make_index_sequence<sizeof...(Ts)>{}, tuple);
        }

        explicit scoped_var(Args const&... args)
            : message_(args...)  //
        {
            scope_logger->log(spdlog::level::info, message_);
        }

        ~scoped_var() { scope_logger->log(spdlog::level::info, message_); }
    };

}  // namespace ippl::debug

#define SPDLOG_SCOPE(...) ippl::debug::scoped_var scope(__VA_ARGS__);

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
