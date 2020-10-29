#include "Ippl.h"
#include <random>

template<class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout>
{

    Bunch(PLayout& playout)
    : ippl::ParticleBase<PLayout>(playout)
    {
        this->addAttribute(E);
    }

    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type E;
};

int main(int argc, char *argv[]) {
    Ippl ippl(argc, argv);

    typedef ippl::detail::ParticleLayout<double, 3> playout_type;
    typedef Bunch<playout_type> bunch_type;

    playout_type pl;

    bunch_type bunch(pl);


    int n = 10;

    bunch.create(n);


    std::mt19937_64 eng;
    std::uniform_real_distribution<double> unif(0, 1);

    typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
    typename bunch_type::charge_container_type::HostMirror E_host = bunch.E.getHostMirror();
    for(int i = 0; i < n; ++i) {
        ippl::Vector<double, 3> r = {unif(eng), unif(eng), unif(eng)};
        R_host(i) = r;
        E_host(i) = 0.0;
    }
    Kokkos::deep_copy(bunch.R.getView(), R_host);
    Kokkos::deep_copy(bunch.E.getView(), E_host);


    int pt = 20;
    Index I(pt);
    NDIndex<3> owned(I, I, I);

    e_dim_tag allParallel[3];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<3; d++)
        allParallel[d] = SERIAL;

    // all parallel layout, standard domain, normal axis order
    FieldLayout<3> layout(owned,allParallel, 1);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    typedef ippl::Field<double, 3> field_type;

    field_type field;

    field.initialize(mesh, layout);

    field = 1.0;


    gather(bunch.E, field, bunch.R);

    bunch.E.print();

    return 0;
}
