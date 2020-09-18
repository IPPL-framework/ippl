#include "Ippl.h"

#include <iostream>
#include <typeinfo>

int main(int argc, char *argv[]) {


  Ippl ippl(argc,argv);
//   Inform testmsg(argv[0], INFORM_ALL_NODES);

    constexpr unsigned int dim = 1;
    typedef Kokkos::View<double*> view_type;


    Index I(4);
    NDIndex<dim> owned(I);
    NDIndex<dim> allocated(I);

    LField<view_type, dim> lfield_1d(owned, allocated);

    view_type* data = lfield_1d.getP();

    std::cout << typeid(*data).name() << std::endl;


    /*
     * Kokkos based LField
     */

    std::cout << "1-dimensional:" << std::endl;
    typedef Kokkos_LField<double, dim> KLField_t;
    KLField_t klfield_1d(owned, allocated);


    klfield_1d.resize(4);

    klfield_1d.write();


    std::cout << "2-dimensional:" << std::endl;
    Index J(4);
    NDIndex<2> owned2d(I, J);
    NDIndex<2> allocated2d(I, J);

    typedef Kokkos_LField<double, 2> kl2_t;
    kl2_t klfield_2d(owned2d, allocated2d);

    klfield_2d.resize(4, 4);

    klfield_2d.write();

    std::cout << "3-dimensional:" << std::endl;
    Index K(4);
    NDIndex<3> owned3d(I, J, K);
    NDIndex<3> allocated3d(I, J, K);

    typedef Kokkos_LField<double, 3> kl3_t;
    kl3_t klfield_3d(owned3d, allocated3d);

    klfield_3d.resize(4, 4, 4);

    klfield_3d.write();
  return 0;
}