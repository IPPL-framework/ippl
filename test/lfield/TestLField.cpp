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
  }
  Kokkos::finalize();

  return 0;
}