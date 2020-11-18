#include <random>

#include "Ippl.h"

int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);

    typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
    typedef ippl::ParticleBase<playout_type> bunch_type;

    constexpr unsigned int dim = 3;

    int pt = 10;
    ippl::Index I(pt);
    NDIndex<dim> owned(I, I, I);

    ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = ippl::PARALLEL;

    ippl::FieldLayout<dim> layout(owned,allParallel);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    playout_type pl(layout, mesh);

    bunch_type bunch(pl);

    using BC = ippl::BC;

    bunch_type::bc_container_type bcs = {
        BC::PERIODIC,
        BC::PERIODIC,
        BC::PERIODIC,
        BC::PERIODIC,
        BC::PERIODIC,
        BC::PERIODIC
    };

    bunch.setParticleBC(bcs);

    int nRanks = Ippl::Comm->size();
    int nParticles = 12;

    if (nParticles % nRanks > 0) {
        if (Ippl::Comm->rank() == 0) {
            std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
        }
        return 0;
    }

    bunch.create(nParticles / nRanks);

    std::mt19937_64 eng(Ippl::Comm->rank());
    std::uniform_real_distribution<double> unif(0, 1);

    typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
    for (size_t i = 0; i < bunch.getLocalNum(); ++i) {
        ippl::Vector<double, dim> r = {unif(eng), unif(eng), unif(eng)};
        R_host(i) = r;
        std::cout << R_host(i) << std::endl;
    }

    Kokkos::deep_copy(bunch.R.getView(), R_host);

    std::cout << Ippl::Comm->rank() << " " << bunch.getLocalNum() << std::endl;

    bunch.update();

    std::cout << "After update:" << std::endl;

    std::cout << Ippl::Comm->rank() << " " << bunch.getLocalNum() << std::endl;


    Kokkos::deep_copy(R_host, bunch.R.getView());

    for (size_t i = 0; i < bunch.getLocalNum(); ++i) {
        std::cout << R_host(i) << std::endl;
    }

    return 0;
}
