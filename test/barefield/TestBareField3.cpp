#include "Ippl.h"

#include <iostream>
#include <typeinfo>

// #include <boost/core/demangle.hpp>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 3;

    ippl::Index I(10);
    NDIndex<dim> owned(I, I, I);


    e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
    for (unsigned int d=0; d<dim; d++)
        allParallel[d] = SERIAL;

    // all parallel layout, standard domain, normal axis order
    FieldLayout<dim> layout(owned, allParallel);

    typedef ippl::BareField<double, dim> bfield_t;
    bfield_t barefield(layout);

    double pi = acos(-1.0);

    barefield = pi/4;

    //barefield.write();

    barefield = fabs(7.0 * (sin(barefield) * cos(barefield))/(tan(barefield) * acos(barefield)) 
                - exp(barefield) + erf(barefield) + (asin(barefield) * cosh(barefield)) / (atan(barefield) 
                * sinh(barefield)) + tanh(barefield) * log(barefield)
               - log10(barefield) * sqrt(barefield) + floor(barefield) * ceil(barefield));

    barefield.write();

    barefield = -barefield;

    barefield.write();


//     const char* name = typeid(2 * barefield + barefield).name();
//     std::cout << boost::core::demangle(name) << std::endl;

    return 0;
}

