// Free Electron Laser simulation.
//
//   Usage:
//     srun ./FreeElectronLaser [<config.json>] --info [0-5]
//    or
//    mpirun -np [N] ./FreeElectronLaser  [<config.json>] --info [0-5]
//
//   Reads a MITHRA-style JSON job file (by default the config.json staged next to
//   the build-tree executable) describing the grid, the relativistic electron
//   bunch, and the undulator. The simulation
//   runs in a Lorentz frame co-moving with the bunch: a charge-conserving
//   current is deposited onto the grid, Maxwell's equations are advanced with a
//   standard FDTD solver (absorbing boundaries), and the particles are pushed
//   with a relativistic Boris pusher that also feels the (frame-transformed)
//   undulator field. Radiated power is written to a CSV and a
//   narrow-band radiation diagnostic is produced alongside it.



    // "resolution": [96, 96, 3000],
    // "resolution": [48, 48, 1500],

constexpr unsigned Dim = 3;
using T                = double;

#include "Ippl.h"

#include <Kokkos_Core.hpp>
#include <string>

#include "Utility/IpplTimings.h"

#include "FreeElectronLaserManager.h"

#ifndef IPPL_FEL_DEFAULT_CONFIG
#define IPPL_FEL_DEFAULT_CONFIG "config.json"
#endif

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("FreeElectronLaser");

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        // First positional argument (if any, and not an --option) overrides the
        // example configuration staged next to the build-tree executable.
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
