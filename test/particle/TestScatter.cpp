#include "Ippl.h"

#include <random>

template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
    }

    ~Bunch() {}

    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef Bunch<playout_type> bunch_type;
        using Mesh_t      = ippl::UniformCartesian<double, 3>;
        using Centering_t = Mesh_t::DefaultCentering;

        int pt = 512;
        ippl::Index I(pt);
        ippl::NDIndex<3> owned(I, I, I);

        std::array<bool, 3> isParallel;
        isParallel.fill(true);

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<3> layout(MPI_COMM_WORLD, owned, isParallel);

        double dx                      = 1.0 / double(pt);
        ippl::Vector<double, 3> hx     = {dx, dx, dx};
        ippl::Vector<double, 3> origin = {0, 0, 0};
        Mesh_t mesh(owned, hx, origin);

        playout_type pl(layout, mesh);

        bunch_type bunch(pl);
        typedef ippl::Field<double, 3, Mesh_t, Centering_t> field_type;

        field_type field;

        field.initialize(mesh, layout);

        bunch.setParticleBC(ippl::BC::PERIODIC);

        int nRanks              = ippl::Comm->size();
        unsigned int nParticles = std::pow(256, 3);

        if (nParticles % nRanks > 0) {
            if (ippl::Comm->rank() == 0) {
                std::cerr << nParticles << " not a multiple of " << nRanks << std::endl;
            }
            return 0;
        }

        unsigned int nLoc = nParticles / nRanks;

        bunch.create(nLoc);

        std::mt19937_64 eng;
        eng.seed(42);
        eng.discard(nLoc * ippl::Comm->rank());
        std::uniform_real_distribution<double> unif(hx[0] / 2, 1 - (hx[0] / 2));

        typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
        double sum_coord                                               = 0.0;
        for (unsigned int i = 0; i < nLoc; ++i) {
            ippl::Vector<double, 3> r = {unif(eng), unif(eng), unif(eng)};
            R_host(i)                 = r;
            sum_coord += r[0] + r[1] + r[2];
        }
        Kokkos::deep_copy(bunch.R.getView(), R_host);

        double global_sum_coord = 0.0;
        ippl::Comm->reduce(sum_coord, global_sum_coord, 1, std::plus<double>());

        if (ippl::Comm->rank() == 0) {
            std::cout << "Sum coord: " << global_sum_coord << std::endl;
        }

        bunch.Q = 1.0;

        bunch.update();

        field = 0.0;

        scatter(bunch.Q, field, bunch.R);

        // Check charge conservation
        try {
            double Total_charge_field = field.sum();

            std::cout << "Total charge in the field:" << Total_charge_field << std::endl;
            std::cout << "Total charge of the particles:" << bunch.Q.sum() << std::endl;
            std::cout << "Error:" << std::fabs(bunch.Q.sum() - Total_charge_field) << std::endl;
        } catch (const std::exception& e) {
            std::cout << e.what() << std::endl;
        }
    }
    ippl::finalize();

    return 0;
}
