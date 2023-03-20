#include "Ippl.h"

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);

    typedef ippl::PRegion<double> region_t;
    typedef ippl::NDRegion<double, 3> ndi_t;

    region_t region(0.1, 0.2);
    ndi_t nd(region, region, region);
    /*
    Kokkos::parallel_for("",
                         1000,
                            KOKKOS_LAMBDA(const int i) {

                         });
*/

    std::cout << nd << std::endl;
    return 0;
}
