#include "Ippl.h"

#include "Random/InverseTransformSampling.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        using Mesh_t = ippl::UniformCartesian<double, 3>;

        ippl::Vector<int, 3> nr   = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};
        const unsigned int ntotal = std::atol(argv[4]);

        ippl::NDIndex<3> domain;
        for (unsigned i = 0; i < 3; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[3];
        for (unsigned d = 0; d < 3; ++d) {
            decomp[d] = ippl::PARALLEL;
        }

        // create mesh and layout objects for this problem domain
        ippl::Vector<double, 3> rmin   = 0;
        ippl::Vector<double, 3> rmax   = 20;
        ippl::Vector<double, 3> length = rmax - rmin;

        ippl::Vector<double, 3> hr     = length / nr;
        ippl::Vector<double, 3> origin = rmin;

        const bool isAllPeriodic = true;
        Mesh_t mesh(domain, hr, origin);
        ippl::FieldLayout<3> fl(domain, decomp, isAllPeriodic);

        ippl::detail::RegionLayout<double, 3, Mesh_t> rlayout(fl, mesh);

        using InvTransSampl_t = ippl::random::InverseTransformSampling<double, 3, Kokkos::Serial>;

        using normal_t = ippl::random::mpi_normal_distribution<Kokkos::Serial>;

        normal_t dist[3];

        ippl::Vector<double, 3> mu, sd;
        mu    = 0.5 * length + origin;
        sd[0] = 0.15 * length[0];
        sd[1] = 0.05 * length[1];
        sd[2] = 0.20 * length[2];

        dist[0] = normal_t(mu[0], sd[0]);
        dist[1] = normal_t(mu[1], sd[1]);
        dist[2] = normal_t(mu[2], sd[2]);

        InvTransSampl_t its(rmin, rmax, rlayout, dist, ntotal, 42);
    }
    ippl::finalize();

    return 0;
}
