// Forward Semi-Lagrangian Test
// Usage:
//   srun ./FSL <nx> <ny> <Np> <Nt> <stype> <dump_freq> --overallocate 1.0 --info 5

constexpr unsigned Dim = 2;
using T                = double;
const char* TestName   = "FSL";

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
#include "FslManager.h"
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
        int nt = std::atoi(argv[arg++]);

        std::string solver = argv[arg++];
        int dump_freq = std::atoi(argv[arg++]);

        msg << " Grid size: " << nr
            << " No. of initial VIC particles: " << np
            << " No. of virtual particles per step: " << nr[0] * nr[1]
            << " No. of time steps: " << nt << endl;

        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax(10.0);
        Vector_t<double, Dim> origin = rmin;
        Vector_t<double, Dim> hr = (rmax - rmin) / nr;

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        const bool isAllPeriodic = true;

        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);

        FSLManager<T, Dim, Band> manager(
            nt,
            nr,
            np,
            solver,
            dump_freq,
            rmin,
            rmax,
            origin,
            FL,
            mesh
        );

        manager.pre_run();
        manager.run(manager.getNt());

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing_fsl.dat"));
    }

    ippl::finalize();

    return 0;
}
