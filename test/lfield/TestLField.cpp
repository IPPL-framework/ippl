#include "Ippl.h"

#include <iostream>

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);

    Index I(10);
    Index J(10);

    Index K(10);
    NDIndex<3> domain(I, J, K);

    ippl::Kokkos_LField<double, 3> lfield(domain);

    lfield = 1.0;

    lfield.write();

    lfield = 5.0 * ((lfield + lfield) * (lfield + lfield)) / (lfield + lfield + lfield) - lfield - 1.0;

    lfield.write();

    return 0;
}