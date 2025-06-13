// Testing the update functionality of the inverse transform sampling method for Normal Distribution
//     Example:
//     srun ./TestInverseTransformSamplingUpdateBounds --overallocate 2.0 --info 10

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

#include "Utility/IpplTimings.h"

#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"

const int Dim = 2;

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;

using Mesh_t = ippl::UniformCartesian<double, Dim>;

using size_type = ippl::detail::size_type;

using GeneratorPool = typename Kokkos::Random_XorShift64_Pool<>;

void get_boundaries(view_type& Rview, double* rmin, double* rmax) {
    double rmax_loc[Dim];
    double rmin_loc[Dim];

    for (unsigned d = 0; d < Dim; ++d) {
        Kokkos::parallel_reduce(
            "rel max", ippl::getRangePolicy(Rview),
            KOKKOS_LAMBDA(const int i, double& mm) {
                double tmp_vel = Rview(i)[d];
                mm             = tmp_vel > mm ? tmp_vel : mm;
            },
            Kokkos::Max<double>(rmax_loc[d]));

        Kokkos::parallel_reduce(
            "rel min", ippl::getRangePolicy(Rview),
            KOKKOS_LAMBDA(const int i, double& mm) {
                double tmp_vel = Rview(i)[d];
                mm             = tmp_vel < mm ? tmp_vel : mm;
            },
            Kokkos::Min<double>(rmin_loc[d]));
    }
    Kokkos::fence();
    MPI_Allreduce(rmax_loc, rmax, Dim, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
    MPI_Allreduce(rmin_loc, rmin, Dim, MPI_DOUBLE, MPI_MIN, ippl::Comm->getCommunicator());
    ippl::Comm->barrier();
}

void write_minmax(double* rmin1, double* rmax1, double* rmin2, double* rmax2) {
    Inform csvout(NULL, "data/rmin_max_normal_dist.csv", Inform::APPEND);
    csvout.precision(10);
    csvout.setf(std::ios::scientific, std::ios::floatfield);
    csvout << "dim    min(pos)    max(pos)    min(vel)    max(vel)" << endl;
    for (int i = 0; i < Dim; i++) {
        csvout << i << " " << rmin1[i] << " " << rmax1[i] << " " << rmin2[i] << " " << rmax2[i]
               << endl;
    }
    ippl::Comm->barrier();
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform m("test ITS normal");
        // set up ITS as other examples to sample position
        ippl::Vector<int, 2> nr = {100, 100};
        size_type ntotal        = 100000;

        ippl::NDIndex<2> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        ippl::Vector<double, Dim> rmin   = -2.;
        ippl::Vector<double, Dim> rmax   = 2.;
        ippl::Vector<double, Dim> length = rmax - rmin;
        ippl::Vector<double, Dim> hr     = length / nr;
        ippl::Vector<double, Dim> origin = rmin;

        const bool isAllPeriodic = true;

        Mesh_t mesh(domain, hr, origin);

        ippl::FieldLayout<Dim> fl(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);

        ippl::detail::RegionLayout<double, Dim, Mesh_t> rlayout(fl, mesh);

        int seed = 42;

        GeneratorPool rand_pool64((size_type)(seed + 100 * ippl::Comm->rank()));

        const double mu1    = 0.1;
        const double sd1    = 0.5;
        const double mu2    = -0.1;
        const double sd2    = 1.0;
        const double par[4] = {mu1, sd1, mu2, sd2};
        using Dist_t        = ippl::random::NormalDistribution<double, Dim>;
        using sampling_t =
            ippl::random::InverseTransformSampling<double, Dim, Kokkos::DefaultExecutionSpace,
                                                   Dist_t>;

        Dist_t dist(par);
        sampling_t sampling(dist, rmax, rmin, rlayout, ntotal);
        size_type nlocal = sampling.getLocalSamplesNum();
        view_type position("position", nlocal);
        sampling.generate(position, rand_pool64);

        // now, we want to sample velocity with the same density, but different bounds
        // update bounds, and related parameters
        ippl::Vector<double, Dim> vmin = 0.;
        ippl::Vector<double, Dim> vmax = 2.;
        sampling.updateBounds(vmax, vmin);
        // set nlocal
        sampling.setLocalSamplesNum(nlocal);
        // create samples of updated ITS
        view_type velocity("velocity", nlocal);
        sampling.generate(velocity, rand_pool64);

        double minPosition[3], maxPosition[3];
        double minVelocity[3], maxVelocity[3];
        get_boundaries(position, minPosition, maxPosition);
        get_boundaries(velocity, minVelocity, maxVelocity);
        write_minmax(minPosition, maxPosition, minVelocity, maxVelocity);
    }
    ippl::finalize();
    return 0;
}
