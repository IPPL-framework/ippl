constexpr unsigned Dim = 3;
using T                = double;
const char* TestName   = "LandauDampingNeurOp";

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <pybind11/embed.h>

#include "Manager/datatypes.h"

#include "Utility/IpplTimings.h"

#include "LandauDampingNeurOpManager.h"

#include "Manager/PicManager.h"

namespace py = pybind11;

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    ippl::initialize(argc, argv);
    int exit_code = 0;
    {
        try {
            Inform msg(TestName);

            static IpplTimings::TimerRef mainTimer       = IpplTimings::getTimer("total");
            static IpplTimings::TimerRef initializeTimer = IpplTimings::getTimer("initialize");
            IpplTimings::startTimer(mainTimer);
            IpplTimings::startTimer(initializeTimer);

            // Read input parameters and assign them to the corresponding memebers of manager
            int arg = 1;
            Vector_t<int, Dim> nr;
            for (unsigned d = 0; d < Dim; d++) {
                nr[d] = std::atoi(argv[arg++]);
            }

            size_type totalP   = std::atoll(argv[arg++]);
            int nt             = std::atoi(argv[arg++]);
            std::string solver = argv[arg++];

            double lbt              = std::atof(argv[arg++]);
            std::string step_method = argv[arg++];

            std::vector<std::string> preconditioner_params;

            if (solver == "PCG" || solver == "FEM_PRECON") {
                while (arg < argc) {
                    const std::string token = argv[arg];
                    if (token.rfind("--", 0) == 0) {
                        break;
                    }
                    preconditioner_params.push_back(token);
                    ++arg;
                }
            }

            const char* inferDir = std::getenv("IPPL_INFER_DIR");
            if (!inferDir) {
                throw std::runtime_error("IPPL_INFER_DIR not set");
            }

            py::scoped_interpreter guard{};

            ippl::Interoperability::InferenceBridge bridge(inferDir, "ippl_inference");
            LandauDampingNeurOpManager<T, Dim> manager(totalP, nt, nr, lbt, solver,
                                                       step_method, preconditioner_params, &bridge);
            manager.pre_run();

            IpplTimings::stopTimer(initializeTimer);

            manager.setTime(0.0);
            manager.run(manager.getNt());
            bridge.finalize();

            IpplTimings::stopTimer(mainTimer);
            IpplTimings::print();
            IpplTimings::print(std::string("timing.dat"));
        } catch (py::error_already_set& e) {
            e.restore();
            PyErr_Print();
            fflush(stderr);
            ippl::Comm->abort();
        } catch (const IpplException& ex) {
            Inform err(TestName);
            err << "IPPL exception: " << ex.what() << endl;
            exit_code = 1;
        } catch (const std::exception& ex) {
            Inform err(TestName, INFORM_ALL_NODES);
            err << "exception: " << ex.what() << endl;
            exit_code = 1;
        } catch (...) {
            Inform err(TestName);
            err << "Unhandled unknown exception" << endl;
            exit_code = 1;
        }
    }
    ippl::finalize();
    return exit_code;
}