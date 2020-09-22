#include "Ippl.h"

#include <iostream>
#include <typeinfo>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 1;

    Index I(16);
    NDIndex<dim> owned(I);
    NDIndex<dim> allocated(I);



    e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = SERIAL;

    // all parallel layout, standard domain, normal axis order
    FieldLayout<dim> layout(owned,allParallel, 1);

    typedef Kokkos_BareField<double, dim> bfield_t;
    bfield_t barefield(layout);

    barefield.write();

    barefield = 1.0;

    barefield.write();

    //
    bfield_t barefield_2(layout);

    barefield_2 = 2.0;

    barefield = barefield_2;

    barefield.write();

    return 0;
}