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
        allParallel[d] = ippl::SERIAL;

    ippl::FieldLayout<dim> layout(owned,allParallel);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    playout_type pl(layout, mesh);

    bunch_type bunch(pl);

    std::cout << bunch.getLocalNum() << std::endl;

    bunch.create(100);

    bunch.update();

    bunch.destroy();

    std::cout << bunch.getLocalNum() << std::endl;

    return 0;
}
