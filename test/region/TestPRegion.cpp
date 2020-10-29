#include "Ippl.h"

#include <typeinfo>

#include <boost/core/demangle.hpp>

int main(int argc, char *argv[]) {

    Ippl ippl(argc,argv);

    typedef ippl::PRegion<double> region_t;

    Kokkos::parallel_for("",
			 1000,
			 KOKKOS_LAMBDA(const int /*i*/) {
			     region_t region(1.0, 2.0);


			     region += 1.0;
			 });

    //    const char* name = typeid(a * x * y * a).name();

    //std::cout << z << std::endl;
    //std::cout << boost::core::demangle( name ) << std::endl;

    return 0;
}
