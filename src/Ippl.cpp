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

namespace ippl {

    void initialize(int& argc, char* argv[], MPI_Comm comm) {
        Comm = std::make_unique<ippl::Communicate>(argc, argv, comm);

        Info  = std::make_unique<Inform>("Ippl");
        Warn  = std::make_unique<Inform>("Warning", std::cerr);
        Error = std::make_unique<Inform>("Error", std::cerr, INFORM_ALL_NODES);

        try {
            std::list<std::string> notparsed;
            int infoLevel = 0;
            int nargs     = 0;
            while (nargs < argc) {
                if (detail::checkOption(argv[nargs], "--help", "-h")) {
                    if (Comm->myNode() == 0) {
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
                } else if (nargs > 0 && std::strstr(argv[nargs], "--kokkos") == nullptr) {
                    notparsed.push_back(argv[nargs]);
                }
                ++nargs;
            }

            Info->setOutputLevel(infoLevel);
            Error->setOutputLevel(0);
            Warn->setOutputLevel(0);

            if (infoLevel > 0 && Comm->myNode() == 0) {
                for (auto& l : notparsed) {
                    std::cout << "Warning: Option '" << l << "' is not parsed by Ippl."
                              << std::endl;
                }
            }
        } catch (const std::exception& e) {
            if (Comm->myNode() == 0) {
                std::cerr << e.what() << std::endl;
            }
            std::exit(0);
        }

        Kokkos::initialize(argc, argv);
    }

    void finalize() {
        Comm->deleteAllBuffers();
        Kokkos::finalize();
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
