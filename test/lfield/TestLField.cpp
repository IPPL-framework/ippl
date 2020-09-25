#include "Ippl.h"

#include <iostream>

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);

//     constexpr unsigned int dim = 1;

    Index I(4);
    NDIndex<1> domain_1d(I);

    /*
     *
     * 1-dimensional
     *
     */
    std::cout << "1-dimensional" << std::endl;
    Kokkos_LField<double, 1> lfield_1d(domain_1d);

    lfield_1d = 1.0;

    lfield_1d.write();

    lfield_1d = ((lfield_1d + lfield_1d) * (lfield_1d + lfield_1d)) / (lfield_1d + lfield_1d + lfield_1d) - lfield_1d;

    lfield_1d.write();

    /*
     *
     * 2-dimensional
     *
     */
    std::cout << "2-dimensional" << std::endl;
    Index J(4);
    NDIndex<2> domain_2d(I, J);

    Kokkos_LField<double, 2> lfield_2d(domain_2d);

    lfield_2d = 1.0;

    lfield_2d.write();

    lfield_2d = ((lfield_2d + lfield_2d) * (lfield_2d + lfield_2d)) / (lfield_2d + lfield_2d + lfield_2d) - lfield_2d;

    lfield_2d.write();

    /*
     *
     * 3-dimensional
     *
     */
    std::cout << "3-dimensional" << std::endl;
    Index K(4);
    NDIndex<3> domain_3d(I, J, K);

    Kokkos_LField<double, 3> lfield_3d(domain_3d);

    lfield_3d = 1.0;

    lfield_3d.write();

    lfield_3d = ((lfield_3d + lfield_3d) * (lfield_3d + lfield_3d)) / (lfield_3d + lfield_3d + lfield_3d) - lfield_3d;

    lfield_3d.write();

    return 0;
}