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
    typedef ippl::BareField<vector_t, dim> bfield_t;
    typedef ippl::BareField<double, dim> bfield_s_t;
    bfield_t barefield(layout);
    bfield_s_t barefield_s(layout);

    barefield = 1.0;

    barefield.write();

    //barefield = 2 * ((barefield + barefield) * (barefield + barefield)) / (barefield + barefield + barefield) - barefield;

    //barefield = 5.0 * cross(barefield, barefield);

    barefield_s = 5.0 * dot(barefield, barefield);

    barefield_s.write();

    return 0;
}
