/** @file FreeElectronLaser.cpp
 * @brief Entry point for the three-dimensional, moving-frame FEL PIC mini-app.
 * @ingroup fel_runtime
 *
 * Reads a MITHRA-style JSON job file, builds the distributed simulation through
 * FreeElectronLaserManager, and runs its prescribed number of steps. The active
 * solver advances collocated four-potentials with the nonstandard vacuum stencil;
 * a relativistic Boris pusher also samples a frame-transformed static undulator.
 * Current deposition, initialization transients and diagnostics have validation
 * limits described in the student guide; this executable does not model media.
 *
 * Usage: `mpirun -np N ./FreeElectronLaser /path/to/config.json --info 5`.
 * Without a positional path, the fallback `config.json` is relative to the
 * process working directory. CMake stages a copy beside the executable, but does
 * not make the executable search that directory automatically.
 */

constexpr unsigned Dim = 3; ///< Spatial dimension; the application assumes beam motion along z.
using T                = double; ///< Floating-point type for particles, fields and manager.

#include "Ippl.h"

#include <Kokkos_Core.hpp>
#include <string>

#include "Utility/IpplTimings.h"

#include "FreeElectronLaserManager.h"

#ifndef IPPL_FEL_DEFAULT_CONFIG
/// Default job-file path; relative paths resolve from the working directory.
#define IPPL_FEL_DEFAULT_CONFIG "config.json"
#endif

/** @brief Initialize IPPL, run the FEL manager, write timings and finalize IPPL.
 * @param argc Number of command-line arguments; IPPL initialization may consume options.
 * @param argv Command-line arguments; the remaining first non-option argument at
 * index 1 selects the JSON file. Later positional arguments are not searched.
 * @return Zero on normal completion. Configuration/runtime exceptions are not caught here.
 *
 * The inner scope destroys fields, particles and their Kokkos storage before
 * ippl::finalize(). Optional Catalyst finalization occurs after the last step.
 * Timing output `timing.dat` is written in the working directory, independently
 * of the configured diagnostic output directory.
 */
int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("FreeElectronLaser");

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        // Only argv[1] is inspected for a job-file override after initialization.
        std::string config_path = IPPL_FEL_DEFAULT_CONFIG;
        if (argc > 1 && argv[1][0] != '-') {
            config_path = argv[1];
        }
        msg << "Reading configuration from " << config_path << endl;
        config cfg = read_config(config_path.c_str());

        // Create the manager for the FEL application.
        FreeElectronLaserManager<T, Dim> manager(cfg);

        // Pre-run: build mesh, fields, particles, FDTD solver; derive dt and nt.
        manager.pre_run();

        msg << "Starting iterations ..." << endl;

        manager.run(manager.getNt());

#ifdef IPPL_ENABLE_CATALYST
        manager.cat_viz.Finalize();
#endif

        msg << "End." << endl;

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
