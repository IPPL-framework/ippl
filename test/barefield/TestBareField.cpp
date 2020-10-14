#include "Ippl.h"

#include <iostream>
#include <typeinfo>

// #include <boost/core/demangle.hpp>

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

    typedef ippl::BareField<double, dim> bfield_t;
    bfield_t barefield(layout);

    barefield = 1.0;

    barefield.write();

    barefield = 2 * ((barefield + barefield) * (barefield + barefield))
              / (barefield + 5.0 * barefield + barefield) - barefield;

    barefield.write();


//     const char* name = typeid(2 * barefield + barefield).name();
//     std::cout << boost::core::demangle(name) << std::endl;

    return 0;
}

