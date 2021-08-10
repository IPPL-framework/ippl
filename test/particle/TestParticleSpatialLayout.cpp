#include <random>

#include "Ippl.h"

template<class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout>
{

    Bunch(PLayout& playout)
    : ippl::ParticleBase<PLayout>(playout)
    {
    }

    ~Bunch(){ }

};

int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);

    Ippl::Comm->setDefaultOverallocation(1);

    typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
    typedef Bunch<playout_type> bunch_type;

    constexpr unsigned int dim = 3;

    int pt = 8;
    ippl::Index I(pt);
    ippl::NDIndex<dim> owned(I, I, I);

    ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = ippl::PARALLEL;

    ippl::FieldLayout<dim> layout(owned,allParallel);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    typedef ippl::UniformCartesian<double, 3> Mesh_t;
    Mesh_t mesh(owned, hx, origin);

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
    unsigned int nParticles = (std::pow(pt, 3))*40;

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
    }

    Ippl::Comm->barrier();
    Kokkos::deep_copy(bunch.R.getView(), R_host);

    if (Ippl::Comm->rank() == 0) {
        std::cout << "Before update:" << std::endl;
    }

    std::cout << layout << std::endl;
    bunch_type bunchBuffer(pl);
    bunchBuffer.create(100);

    pl.update(bunch, bunchBuffer);

    Ippl::Comm->barrier();


    if (Ippl::Comm->rank() == 0) {
        std::cout << "After update:" << std::endl;
    }

    unsigned int Total_particles = 0;
    unsigned int local_particles = bunch.getLocalNum();

    MPI_Reduce(&local_particles, &Total_particles, 1, 
                MPI_UNSIGNED, MPI_SUM, 0, Ippl::getComm());
    if (Ippl::Comm->rank() == 0) {
        
        std::cout << "Total particles before: " << nParticles 
                  << " " << "after: " << Total_particles << std::endl;
    }
    return 0;
}
