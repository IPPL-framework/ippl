#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <array>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;

    std::array<int, dim> pt = {8, 7, 13};
    ippl::Index I(pt[0]);
    ippl::Index J(pt[1]);
    ippl::Index K(pt[2]);
    ippl::NDIndex<dim> owned(I, J, K);

    ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = ippl::PARALLEL;

    ippl::FieldLayout<dim> layout(owned, allParallel);

    std::array<double, dim> dx = {
        1.0 / double(pt[0]),
        1.0 / double(pt[1]),
        1.0 / double(pt[2]),
    };
    ippl::Vector<double, 3> hx = {dx[0], dx[1], dx[2]};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);


    typedef ippl::Field<double, dim> field_type;

    field_type field(mesh, layout, 2);

    field = 1.0;

//     field.write();

//     field.exchangeHalo();

    std::cout << std::endl;

    field.fillLocalHalo(2.0);

    field.write();

    return 0;
}
