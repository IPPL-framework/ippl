// Vortex In Cell Test
//   Usage:
//     srun ./VortexInCell
//                  <nx> [<ny>...] <Np> <Nt> <stype> <lbthres>
//                  <t_method> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     visc     = Viscosity
//     stype    = Field solver type (FFT and CG supported)
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     t_method = Time-stepping method used e.g. Leapfrog
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     makdir build_*/alvine/data
//     chmod +x data
//     srun ./VortexInCell 128 128 100 FFT --overallocate 2.0 --info 10
//     srun ./VortexInCell 128 128
//     to build, call 
//          make VortexInCell 
//     in the build directory to only build this target

constexpr unsigned Dim = 2;
using T                = double;
const char* TestName   = "VortexInCell";

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

#include "datatypes.h"

#include "Utility/IpplTimings.h"

#include "Manager/PicManager.h"
#include "VortexInCellManager.h"
#include "VortexDistributions.h"
#include "ParticleDistributions.h"


int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        unsigned arg = 1;    
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        }

        int nt  = std::atoi(argv[arg++]);

        std::string solver = argv[arg++];

        double lbt = std::atof(argv[arg++]);

        msg << nt << endl;

        VortexInCellManager<T, Dim, EquidistantDistribution, ConcentricCircles> manager(nt, nr, solver, lbt);

        manager.pre_run();

        manager.run(manager.getNt());

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
