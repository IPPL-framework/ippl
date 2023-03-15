#include "Ippl.h"

#include <Kokkos_Core.hpp>

#include <initializer_list>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int length = 10;

        typedef ippl::Vector<double, 3> vector_type;

        typedef Kokkos::View<vector_type***> vector_field_type;

        vector_field_type vfield("vfield", length, length, length);

        using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

        Kokkos::parallel_for(
            "assign", mdrange_t({0, 0, 0}, {length, length, length}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                vfield(i, j, k) = {1.0, 2.0, 3.0};
            });

        vector_field_type wfield("wfield", length, length, length);

        Kokkos::parallel_for(
            "assign", mdrange_t({0, 0, 0}, {length, length, length}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                wfield(i, j, k) = {4.0, -5.0, 6.0};
            });

        vector_field_type vvfield("vvfield", length, length, length);
        Kokkos::parallel_for(
            "assign", mdrange_t({0, 0, 0}, {length, length, length}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                vvfield(i, j, k) = 0.25 * wfield(i, j, k)
                                       * cross(vfield(i, j, k) * wfield(i, j, k), wfield(i, j, k))
                                   + wfield(i, j, k) / vfield(i, j, k) + 2.0;
            });

        Kokkos::fence();

        vector_field_type::HostMirror host_view = Kokkos::create_mirror_view(vvfield);
        Kokkos::deep_copy(host_view, vvfield);

        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                for (int k = 0; k < length; ++k) {
                    std::cout << host_view(i, j, k)[0] << " " << host_view(i, j, k)[1] << " "
                              << host_view(i, j, k)[2] << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        typedef Kokkos::View<double***> scalar_field_type;
        scalar_field_type sfield("sfield", length, length, length);

        Kokkos::parallel_for(
            "assign", mdrange_t({0, 0, 0}, {length, length, length}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                sfield(i, j, k) = dot(wfield(i, j, k), wfield(i, j, k)).apply();
            });

        Kokkos::fence();

        scalar_field_type::HostMirror host_sview = Kokkos::create_mirror_view(sfield);
        Kokkos::deep_copy(host_sview, sfield);

        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                for (int k = 0; k < length; ++k) {
                    std::cout << host_sview(i, j, k) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
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
