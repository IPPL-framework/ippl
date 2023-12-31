#include "Ippl.h"

#include <typeinfo>

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        typedef ippl::PRegion<double> region_t;

        Kokkos::parallel_for(
            "", 1000, KOKKOS_LAMBDA(const int /*i*/) {
                region_t region(1.0, 2.0);

                region += 1.0;
            });
    }
    ippl::finalize();

    return 0;
}
