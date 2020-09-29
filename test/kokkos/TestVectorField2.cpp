#include "Ippl.h"

#include <Kokkos_Core.hpp>

#include <iostream>


int main(int argc, char *argv[]) {

    Kokkos::initialize(argc,argv);
    {
        constexpr int length = 10;

        typedef Vektor<double, 3> vector_type;

        typedef Kokkos::View<vector_type*> vector_field_type;

        vector_field_type vfield("vfield", length);

        Kokkos::parallel_for("assign", length, KOKKOS_LAMBDA(const int i) {
            vfield(i) = vector_type(2.0);
        });

        Kokkos::fence();

        vector_field_type::HostMirror host_view = Kokkos::create_mirror_view(vfield);
        Kokkos::deep_copy(host_view, vfield);


        for (int i = 0; i < length; ++i) {
            std::cout << host_view(i)(0) << " " << host_view(i)(1) << " " << host_view(i)(2) << std::endl;
        }
    }
    Kokkos::finalize();

    return 0;
}