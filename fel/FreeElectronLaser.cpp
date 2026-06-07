// Free Electron Laser simulation.
//
//   Usage:
//     srun ./FreeElectronLaser [<config.json>] --info 5
//
//   Reads a MITHRA-style JSON job file (default: ../config.json) describing the
//   grid, the relativistic electron bunch, and the undulator. The simulation
//   runs in a Lorentz frame co-moving with the bunch: a charge-conserving
//   current is deposited onto the grid, Maxwell's equations are advanced with a
//   standard FDTD solver (absorbing boundaries), and the particles are pushed
//   with a relativistic Boris integrator that also feels the (frame-transformed)
//   undulator field. Radiated power is written to a CSV and, optionally, a
//   Poynting-flux video is produced via ffmpeg.

constexpr unsigned Dim = 3;
using T                = double;

#include "Ippl.h"

#include <Kokkos_Core.hpp>
#include <string>

#include "Utility/IpplTimings.h"

// stb_image_write's implementation must be emitted in exactly one translation
// unit; define the macro before the (transitive) include of the header.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "FreeElectronLaserManager.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("FreeElectronLaser");

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        // First positional argument (if any, and not an --option) is the config
        // file path; otherwise fall back to ../config.json as the original did.
        const char* config_path = "../config.json";
        if (argc > 1 && argv[1][0] != '-') {
            config_path = argv[1];
        }
        msg << "Reading configuration from " << config_path << endl;
        config cfg = read_config(config_path);

        // Create the manager for the FEL application.
        FreeElectronLaserManager<T, Dim> manager(cfg);

        // Pre-run: build mesh, fields, particles, FDTD solver; derive dt and nt.
        manager.pre_run();

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
