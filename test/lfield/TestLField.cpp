#include "Ippl.h"


#include <Kokkos_Core.hpp>

#include <iostream>
#include <typeinfo>

int main(int argc, char *argv[]) {


  Ippl ippl(argc,argv);
//   Inform testmsg(argv[0], INFORM_ALL_NODES);

  Kokkos::initialize(argc, argv);
  {
      constexpr unsigned int dim = 1;
      typedef Kokkos::View<double*> view_type;


      Index I(32);
      NDIndex<dim> owned(I);
      NDIndex<dim> allocated(I);

      LField<view_type, dim> lfield_1d(owned, allocated);

      view_type* data = lfield_1d.getP();

      std::cout << typeid(*data).name() << std::endl;


      /*
       * Kokkos based LField
       */
      typedef Kokkos_LField<double, dim> KLField_t;
      KLField_t klfield_1d(owned, allocated);

      KLField_t::view_type kdata = klfield_1d.getP();

      std::cout << typeid(kdata).name() << std::endl;


      Index J(32);
      NDIndex<2> owned2d(I, J);
      NDIndex<2> allocated2d(I, J);

      typedef Kokkos_LField<double, 2> kl2_t;
      kl2_t klfield_2d(owned2d, allocated2d);

      kl2_t::view_type kdata2d = klfield_2d.getP();

      std::cout << typeid(kdata2d).name() << std::endl;

      Index K(32);
      NDIndex<3> owned3d(I, J, K);
      NDIndex<3> allocated3d(I, J, K);

      typedef Kokkos_LField<double, 3> kl3_t;
      kl3_t klfield_3d(owned3d, allocated3d);

      kl3_t::view_type kdata3d = klfield_3d.getP();

      std::cout << typeid(kdata3d).name() << std::endl;
  }
  Kokkos::finalize();

  return 0;
}