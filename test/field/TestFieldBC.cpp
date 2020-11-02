#include "Ippl.h"

#include <iostream>
#include <typeinfo>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;

    int pt = 8;
    Index I(pt);
    NDIndex<dim> owned(I, I, I);

    e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = SERIAL;

    FieldLayout<dim> layout(owned,allParallel, 1);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};

    using Mesh_t = ippl::UniformCartesian<double, 3>;

    Mesh_t mesh(owned, hx, origin);


    typedef ippl::Field<double, dim> field_type;
//     typedef ippl::Field<ippl::Vector<double, dim>, dim> vector_field_type;

    field_type::bc_container bc;

    // Boundary conditions--zero on all faces:
    bc[0] = std::make_shared<ippl::NoBcFace<double, 3> >(0);
    bc[1] = std::make_shared<ippl::ConstantFace<double, 3> >(1, 0);
    bc[2] = std::make_shared<ippl::ZeroFace<double, 3> >(2);
    bc[3] = std::make_shared<ippl::PeriodicFace<double, 3> >(3);
    bc[4] = std::make_shared<ippl::NoBcFace<double, 3> >(4);
    bc[5] = std::make_shared<ippl::ZeroFace<double, 3> >(5);


    std::cout << bc << std::endl;

    field_type field(mesh, layout, bc);

    field = 1.0;

    return 0;
}