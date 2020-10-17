#include "Ippl.h"
//#include <Kokkos_Core.hpp>

#include <iostream>

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);

    Index I(10);
    Index J(10);

    Index K(10);
    NDIndex<3> domain(I, J, K);

    ippl::LField<double, 3> lfield(domain);

    double pi = acos(-1.0);

    lfield = pi/4;

    //lfield.write();

    lfield = fabs(7.0 * (sin(lfield) * cos(lfield))/(tan(lfield) * acos(lfield)) - exp(lfield) + erf(lfield)
               + (asin(lfield) * cosh(lfield))/(atan(lfield) * sinh(lfield)) + tanh(lfield) * log(lfield)
               - log10(lfield) * sqrt(lfield) + floor(lfield) * ceil(lfield));

    lfield.write();

    lfield = -lfield;

    lfield.write();

    return 0;
}
