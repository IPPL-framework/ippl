constexpr unsigned Dim = 3;
using T                = double;
const char* TestName   = "MovingParticle";

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

template <class PLayout>
struct Bunch: public ippl::ParticleBase<PLayout> {
    ParticleAttrib<Vector<double>> V;

    Bunch(PLayout& layout) : ippl::ParticleBase<PLayout>(layout) {
        this->addAttribute(V);
    }
    ~Bunch() {};
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef Bunch<playout_type> bunch_type;
        using Mesh_t      = ippl::UniformCartesian<double, 3>;
        using Centering_t = Mesh_t::DefaultCentering;

        Inform msg(TestName);

        // create Particle bunch
        int pt = 512;
        ippl::Index I(pt);
        ippl::NDIndex<Dim> owned(I, I, I);

        std::array<bool, Dim> isParrallel;
        isParrallel.fill(false);

        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParrallel);

        double dx = 1.0 / double(pt);
        Vector<double, Dim> hx = {dx, dx, dx};
        Vector<double, Dim> origin = {0.0, 0.0, 0.0};

        Mesh_t mesh(owned, hx, origin);

        playout_type pl(layout, mesh);

        bunch_type bunch(pl);

        int n = 1;
        bunch.create(n);

        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif(0, 1);

        typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
        typename bunch_type::charge_container_type::HostMirror V_host  = bunch.V.getHostMirror();

        for (int i = 0; i < n; ++i) {
            ippl::Vector<double, 3> r = {unif(eng), unif(eng), unif(eng)};
            R_host(i)                 = r;
            V_host(i)                 = 0.0;
        }
    }
    ippl::finalize();
 
    return 0;
}