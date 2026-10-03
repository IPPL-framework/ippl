#include "CosmologySimulation.h"

int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    try {
        if (argc != 2) throw std::invalid_argument("Usage: Cosmology input.par | --self-test");
        {
            if (std::string(argv[1]) == "--self-test") {
                cosmology::Config config;
                config.nGrid = 16;
                cosmology::Simulation simulation(config);
                simulation.testSpectralForce();
                simulation.testMigration();
            } else {
                const auto config = cosmology::Config::fromFile(argv[1]);
                cosmology::Simulation simulation(config);
                simulation.run();
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Cosmology rank " << ippl::Comm->rank() << ": " << error.what() << '\n';
        MPI_Abort(ippl::Comm->getCommunicator(), 1);
        return 1;
    }
    ippl::finalize();
    return 0;
}
