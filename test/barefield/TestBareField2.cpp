#include "Ippl.h"

#include <iostream>
#include <typeinfo>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;

    Index I(10);
    NDIndex<dim> owned(I, I, I);


    e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = SERIAL;

    // all parallel layout, standard domain, normal axis order
    FieldLayout<dim> layout(owned,allParallel, 1);

    typedef ippl::Vector<double, 3> vector_t;
    typedef ippl::Kokkos_BareField<vector_t, dim> bfield_t;
    bfield_t barefield(layout);

    barefield = 1.0;

    barefield.write();

    barefield = 2 * ((barefield + barefield) * (barefield + barefield)) / (barefield + barefield + barefield) - barefield;

    barefield.write();

    return 0;
}
