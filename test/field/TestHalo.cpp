#include "Ippl.h"

#include <iostream>
#include <typeinfo>
#include <array>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;

//     std::array<int, dim> pt = {8, 7, 13};
    std::array<int, dim> pt = {8, 8, 8};
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

    field = 5.0;

//     field.write();

//     layout.findNeighbors(2);


//     field.exchangeHalo();
//
// //     std::cout << std::endl;
//
// //     field.fillLocalHalo(2.0);
//
//     int nRanks = Ippl::Comm->size();
//
//     for (int rank = 0; rank < nRanks; ++rank) {
//         if (rank == Ippl::Comm->rank()) {
//             std::cout << field.getOwned().grow(2) << std::endl;
//             field.write();
//             std::cout << "--------------------------" << std::endl;
//         }
//         Ippl::Comm->barrier();
//     }

    return 0;
}
