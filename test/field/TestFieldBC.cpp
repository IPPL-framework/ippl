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
    for (unsigned int i = 0; i < 2 * dim; ++i) {
        bc[i] = std::make_shared<ippl::ConstantFace<double, 3> >(i, 0.0);
        std::cout << *bc[i] << std::endl;
    }

    field_type field(mesh, layout, bc);

    field = 1.0;

    return 0;
}