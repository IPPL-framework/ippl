#include "Ippl.h"

#include "Random/InverseTransformSampling.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        using Mesh_t = ippl::UniformCartesian<double, 2>;

        ippl::Vector<int, 2> nr   = {std::atoi(argv[1]), std::atoi(argv[2])};
        const unsigned int ntotal = std::atol(argv[3]);

        ippl::NDIndex<2> domain;
        for (unsigned i = 0; i < 2; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[2];
        for (unsigned d = 0; d < 2; ++d) {
            decomp[d] = ippl::PARALLEL;
        }

        // create mesh and layout objects for this problem domain
        ippl::Vector<double, 2> rmin   = 0;
        ippl::Vector<double, 2> rmax   = 20;
        ippl::Vector<double, 2> length = rmax - rmin;

        ippl::Vector<double, 2> hr     = length / nr;
        ippl::Vector<double, 2> origin = rmin;

        const bool isAllPeriodic = true;
        Mesh_t mesh(domain, hr, origin);
        ippl::FieldLayout<2> fl(domain, decomp, isAllPeriodic);

        ippl::detail::RegionLayout<double, 2, Mesh_t> rlayout(fl, mesh);

        using InvTransSampl_t = ippl::random::InverseTransformSampling<double, 2, Kokkos::Serial>;

        using normal_t = ippl::random::mpi::normal_distribution<Kokkos::Serial>;
        normal_t dist[2];

        ippl::Vector<double, 2> mu, sd;
        mu    = 0.5 * length + origin;
        sd[0] = 0.15 * length[0];
        sd[1] = 0.05 * length[1];

        dist[0] = normal_t(mu[0], sd[0]);
        dist[1] = normal_t(mu[1], sd[1]);

        InvTransSampl_t its(rmin, rmax, rlayout, dist, ntotal);

        using view_type = ippl::detail::ViewType<ippl::Vector<double, 2>, 1>::view_type;

        unsigned int nlocal = its.getLocalNum();

        view_type position("position", nlocal);

        its.generate(dist, position, 42);

        for (unsigned int i = 0; i < nlocal; ++i) {
            std::cout << ippl::Comm->rank() << " " << position(i)[0] << " " << position(i)[1]
                      << std::endl;
        }
    }
    ippl::finalize();

    return 0;
}
