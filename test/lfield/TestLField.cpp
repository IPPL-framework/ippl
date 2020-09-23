#include "Ippl.h"

#include <iostream>
#include <typeinfo>

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);

    constexpr unsigned int dim = 1;

    Index I(4);
    NDIndex<dim> domain(I);

    std::cout << "1-dimensional:" << std::endl;
    typedef Kokkos_LField<double, dim> KLField_t;
//     KLField_t klfield_1d(owned, allocated);
//
//
//     klfield_1d.resize(4);
//
//     klfield_1d.write();
//
//
//     std::cout << "2-dimensional:" << std::endl;
//     Index J(4);
//     NDIndex<2> owned2d(I, J);
//     NDIndex<2> allocated2d(I, J);
//
//     typedef Kokkos_LField<double, 2> kl2_t;
//     kl2_t klfield_2d(owned2d, allocated2d);
//
//     klfield_2d.resize(4, 4);
//
//     klfield_2d.write();
//
//     std::cout << "3-dimensional:" << std::endl;
//     Index K(4);
//     NDIndex<3> owned3d(I, J, K);
//     NDIndex<3> allocated3d(I, J, K);
//
//     typedef Kokkos_LField<double, 3> kl3_t;
//     kl3_t klfield_3d(owned3d, allocated3d);
//
//     klfield_3d.resize(4, 4, 4);
//
//     klfield_3d.write();


    /*
     *
     *
     */
    KLField_t klfield_1(domain);
    klfield_1.resize(4);

    klfield_1 = 1.0;

    klfield_1.write();

    KLField_t klfield_2(domain);
    klfield_2.resize(4);

    klfield_2 = 2.0;

    klfield_2.write();

    klfield_1 = (klfield_2 + klfield_2 * klfield_2 * klfield_2) / klfield_2 - klfield_1;

    Kokkos::fence();

    klfield_1.write();


  return 0;
}