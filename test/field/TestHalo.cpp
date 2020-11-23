#include "Ippl.h"

#include <iostream>
#include <typeinfo>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;

    int pt = 4;
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


    typedef ippl::Field<double, dim> field_type;

    field_type field(mesh, layout, 2);

    field = 1.0;

    field.write();

    std::cout << std::endl;

    field.fillLocalHalo(2.0);

    field.write();

    return 0;
}
