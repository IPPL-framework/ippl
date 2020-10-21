#include "Ippl.h"

#include <Kokkos_Core.hpp>

#include <initializer_list>
#include <iostream>

int main(int argc, char *argv[]) {

    Kokkos::initialize(argc,argv);
    {
        constexpr int length = 10;

        typedef ippl::Vector<double, 3> vector_type;

        typedef Kokkos::View<vector_type*> vector_field_type;

        vector_field_type vfield("vfield", length);

        Kokkos::parallel_for("assign", length, KOKKOS_LAMBDA(const int i) {
            vfield(i) = {1.0, 2.0, 3.0};
        });


         vector_field_type wfield("wfield", length);

         Kokkos::parallel_for("assign", length, KOKKOS_LAMBDA(const int i) {
            wfield(i) = {4.0, -5.0, 6.0};
         });


         vector_field_type vvfield("vvfield", length);
         Kokkos::parallel_for("assign", length, KOKKOS_LAMBDA(const int i) {
                 vvfield(i) = 0.25 * wfield(i) * cross(vfield(i) * wfield(i), wfield(i)) + wfield(i) / vfield(i) + 2.0;
         });

        Kokkos::fence();

        vector_field_type::HostMirror host_view = Kokkos::create_mirror_view(vvfield);
        Kokkos::deep_copy(host_view, vvfield);


        for (int i = 0; i < length; ++i) {
            std::cout << host_view(i)[0] << " " << host_view(i)[1] << " " << host_view(i)[2] << std::endl;
        }



        typedef Kokkos::View<double*> scalar_field_type;
        scalar_field_type sfield("sfield", length);


        Kokkos::parallel_for("assign", length, KOKKOS_LAMBDA(const int i) {
            sfield(i) = dot(wfield(i), wfield(i)).apply();
        });

        Kokkos::fence();

        scalar_field_type::HostMirror host_sview = Kokkos::create_mirror_view(sfield);
        Kokkos::deep_copy(host_sview, sfield);


        for (int i = 0; i < length; ++i) {
            std::cout << host_sview(i) << std::endl;
        }

    }
    Kokkos::finalize();

    return 0;
}

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
