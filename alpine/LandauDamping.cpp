// Landau Damping Test
//   Usage:
//     srun ./LandauDamping
//                  <nx> [<ny>...] <Np> <Nt> <stype> <lbthres>
//                  <t_method> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT and CG supported)
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     t_method = Time-stepping method used e.g. Leapfrog
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./LandauDamping 128 128 128 10000 10 FFT 0.01 LeapFrog --overallocate 2.0 --info 10

constexpr unsigned Dim = 3;
using T                = double;
const char* TestName   = "LandauDamping";

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

#include "Manager/datatypes.h"

#include "Utility/IpplTimings.h"

#include "LandauDampingManager.h"
#include "Manager/PicManager.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);
        Inform msg2all(TestName, INFORM_ALL_NODES);

        // Optional warmup: --warmup <N>. Defaults to 0 (no warmup, baseline
        // behaviour). Stripping the flag from argv here lets the positional
        // parser below see the same layout as before.
        int n_warmup = 0;
        {
            int write_i = 1;
            for (int read_i = 1; read_i < argc; ) {
                if (std::string(argv[read_i]) == "--warmup" && read_i + 1 < argc) {
                    n_warmup = std::atoi(argv[read_i + 1]);
                    read_i += 2;
                } else {
                    argv[write_i++] = argv[read_i++];
                }
            }
            argc = write_i;
        }

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef initializeTimer = IpplTimings::getTimer("initialize");
        IpplTimings::startTimer(mainTimer);
        IpplTimings::startTimer(initializeTimer);

        // Read input parameters, assign them to the corresponding memebers of manager
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
            for (int i = 0; i < 5; i++) {
                preconditioner_params.push_back(argv[arg++]);
            }
        }

        // Create an instance of a manager for the considered application
        LandauDampingManager<T, Dim> manager(totalP, nt, nr, lbt, solver, step_method,
                                             preconditioner_params);

        // Perform pre-run operations, including creating mesh, particles,...
        manager.pre_run();

        IpplTimings::stopTimer(initializeTimer);

        manager.setTime(0.0);

        // Optional warmup: run N timesteps before the timed run so JIT,
        // first-touch allocations, GPU caches, MPI/IPC registration and
        // any tile-size autotune transients don't show up in the measured
        // timers. After the warmup we wipe ALL accumulated timer state so
        // the printed report covers exactly the measured run, not the
        // warmup pass. Pass --warmup <N> on the CLI to enable.
        if (n_warmup > 0) {
            msg << "Running " << n_warmup << " warmup step(s) ..." << endl;
            manager.run(n_warmup);

            // Reset simulation state so the measured run starts from t = 0,
            // step 0 — comparable across branches.
            manager.setTime(0.0);
            manager.setIt(0);

            // Wipe all timer accumulators (including 'total' and 'initialize'
            // we already started above) so the report reflects only the run.
            IpplTimings::stopTimer(mainTimer);
            IpplTimings::resetAllTimers();
            IpplTimings::startTimer(mainTimer);
        }

        msg << "Starting iterations ..." << endl;

        manager.run(manager.getNt());

        msg << "End." << endl;

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
