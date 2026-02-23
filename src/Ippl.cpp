//
// Class Ippl
//   Ippl environment.
//
#include <Kokkos_Core.hpp>
#include "Ippl.h"

#include <cstdlib>
#include <cstring>
#include <list>

#include "Utility/IpplInfo.h"

#if defined(IPPL_LOGGING_ENABLED)
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#endif

#if defined(IPPL_LOGGING_ENABLED)
namespace {
    spdlog::level::level_enum default_spdlog_level() {
#if SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_TRACE
        return spdlog::level::trace;
#elif SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_DEBUG
        return spdlog::level::debug;
#elif SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_INFO
        return spdlog::level::info;
#elif SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_WARN
        return spdlog::level::warn;
#elif SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_ERROR
        return spdlog::level::err;
#elif SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_CRITICAL
        return spdlog::level::critical;
#else
        return spdlog::level::info;
#endif
    }

    void initialize_spdlog_from_env() {
        const char* pattern = std::getenv("IPPL_LOG_PATTERN");
        if (pattern == nullptr || pattern[0] == '\0') {
            pattern = std::getenv("SPDLOG_PATTERN");
        }

        if (pattern && pattern[0] != '\0') {
            spdlog::set_pattern(pattern);
        } else {
            spdlog::set_pattern("[%^%-8l%$]%t| %v");
        }

        spdlog::set_level(default_spdlog_level());
        spdlog::cfg::load_env_levels();

        const char* level = std::getenv("IPPL_LOG_LEVEL");
        if (level && level[0] != '\0') {
            spdlog::set_level(spdlog::level::from_str(level));
        }
    }
}  // namespace
#endif

namespace ippl {

    void initialize(int& argc, char* argv[], MPI_Comm comm) {
        Env = std::make_unique<mpi::Environment>(argc, argv, comm);

        Comm = std::make_unique<mpi::Communicator>(comm);

        Info  = std::make_unique<Inform>("Ippl");
        Warn  = std::make_unique<Inform>("Warning", std::cerr);
        Error = std::make_unique<Inform>("Error", std::cerr, INFORM_ALL_NODES);

#if defined(IPPL_LOGGING_ENABLED)
        initialize_spdlog_from_env();
#endif

        try {
            std::list<std::string> notparsed;
            int infoLevel = 0;
            int nargs     = 0;
            while (nargs < argc) {
                if (detail::checkOption(argv[nargs], "--help", "-h")) {
                    if (Comm->rank() == 0) {
                        IpplInfo::printHelp(argv);
                    }
                    std::exit(0);
                } else if (detail::checkOption(argv[nargs], "--info", "-i")) {
                    ++nargs;
                    if (nargs >= argc) {
                        throw std::runtime_error("Missing info level value!");
                    }
                    infoLevel = detail::getNumericalOption<int>(argv[nargs]);
                } else if (detail::checkOption(argv[nargs], "--timer-fences", "")) {
                    ++nargs;
                    if (nargs >= argc) {
                        throw std::runtime_error("Missing timer fence enable option!");
                    }
                    if (std::strcmp(argv[nargs], "on") == 0) {
                        Timer::enableFences = true;
                    } else if (std::strcmp(argv[nargs], "off") == 0) {
                        Timer::enableFences = false;
                    } else {
                        throw std::runtime_error("Invalid timer fence option");
                    }
                } else if (detail::checkOption(argv[nargs], "--version", "-v")) {
                    IpplInfo::printVersion();
                    std::string options = IpplInfo::compileOptions();
                    std::string header("Compile-time options: ");
                    while (options.length() > 58) {
                        std::string line = options.substr(0, 58);
                        size_t n         = line.find_last_of(' ');
                        *Info << header << line.substr(0, n) << "\n";

                        header  = std::string(22, ' ');
                        options = options.substr(n + 1);
                    }
                    *Info << header << options << endl;
                    std::exit(0);
                } else if (detail::checkOption(argv[nargs], "--overallocate", "-b")) {
                    ++nargs;
                    if (nargs >= argc) {
                        throw std::runtime_error("Missing overallocation factor value!");
                    }
                    auto factor = detail::getNumericalOption<double>(argv[nargs]);
                    Comm->setDefaultOverallocation(factor);
                } else if (detail::checkOption(argv[nargs], "--debug", "-g")) {
                    ++nargs;
                    if (Comm->rank() == 0) {
                        std::cout << "Please attach debugger and hit return" << std::endl;
                        char c;
                        std::cin >> c;
                    }
                    Comm->barrier();
                } else if (nargs > 0 && std::strstr(argv[nargs], "--kokkos") == nullptr) {
                    notparsed.push_back(argv[nargs]);
                }
                ++nargs;
            }

            Info->setOutputLevel(infoLevel);
            Error->setOutputLevel(infoLevel);
            Warn->setOutputLevel(infoLevel);

        } catch (const std::exception& e) {
            if (Comm->rank() == 0) {
                std::cerr << e.what() << std::endl;
            }
            std::exit(0);
        }

        Kokkos::initialize(argc, argv);
    }

    void finalize() {
        Comm->deleteAllBuffers();
        Kokkos::finalize();
        // we must first delete the communicator and
        // afterwards the MPI environment
        Comm.reset(nullptr);
        Env.reset(nullptr);
    }

    void fence() {
        Kokkos::fence();
    }

    void abort(const char* msg, int errorcode) {
        if (msg) {
            *Error << msg << endl;
        }
        Comm->abort(errorcode);
    }

    namespace detail {
        bool checkOption(const char* arg, const char* lstr, const char* sstr) {
            return (std::strcmp(arg, lstr) == 0) || (std::strcmp(arg, sstr) == 0);
        }

        template <typename T, typename>
        T getNumericalOption(const char* arg) {
            constexpr bool isInt = std::is_integral_v<T>;
            std::string sarg     = arg;
            try {
                T ret;
                size_t parsed;
                if constexpr (isInt) {
                    ret = std::stoll(sarg, &parsed);
                } else {
                    ret = std::stold(sarg, &parsed);
                }
                if (parsed != sarg.length())
                    throw std::invalid_argument("Failed to parse");
                return ret;
            } catch (std::invalid_argument& e) {
                if constexpr (isInt) {
                    throw std::runtime_error("Expected integer argument!");
                } else {
                    throw std::runtime_error("Expected floating point argument!");
                }
            }
            // Silence nvcc warning: missing return statement at end of non-void function
            throw std::runtime_error("Unreachable state");
        }
    }  // namespace detail
}  // namespace ippl
