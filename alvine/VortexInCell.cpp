// Vortex In Cell Test
//   Usage:
//     srun ./VortexInCell
//                  <nx> [<ny>...] <Np> <Nt> <stype> <dump_freq> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...-direction
//     Np       = No. of vortex particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT and CG supported)
//     dump_freq= Dumping frequency of particle output
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     makdir build_*/alvine/data
//     chmod +x data
//     srun ./VortexInCell 128 128 10000 100 FFT 100 --overallocate 1.0 --info 5
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

        int np = std::atoi(argv[arg++]);
        int nt  = std::atoi(argv[arg++]);
        std::string solver = argv[arg++];
        int dump_freq  = std::atoi(argv[arg++]);
        
        msg << " Grid size: " << nr << " No. of particles: " << np << " No. of time steps: " << nt << endl;
        
        // ===== CRITICAL: Create mesh and layout with proper MPI decomposition =====
        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        // Domain bounds (adjust as needed for your vortex problem)
        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax(10.0);
        Vector_t<double, Dim> origin = rmin;
        Vector_t<double, Dim> hr = (rmax - rmin) / nr;

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        const bool isAllPeriodic = true;

        // Create the mesh and layout with MPI communicator
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);
        
        // Now create manager WITH the layout info
        VortexInCellManager<T, Dim, Band> manager(nt, nr, np, solver, dump_freq, 
                                                   rmin, rmax, origin, FL, mesh);

        manager.pre_run();
        manager.run(manager.getNt());
        
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
