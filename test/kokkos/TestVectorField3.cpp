#include <Kokkos_Core.hpp>
#include "Ippl.h"

#include <initializer_list>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        constexpr int length = 10;

        typedef ippl::Vector<double, 3> vector_type;

        typedef Kokkos::View<vector_type**> vector_field_type;

        vector_field_type vfield("vfield", length, length);

        using mdrange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

        Kokkos::parallel_for(
            "assign", mdrange_t({0, 0}, {length, length}), KOKKOS_LAMBDA(const int i, const int j) {
                vfield(i, j) = {1.0, 2.0, 3.0};
            });

        vector_field_type wfield("wfield", length, length);

        Kokkos::parallel_for(
            "assign", mdrange_t({0, 0}, {length, length}), KOKKOS_LAMBDA(const int i, const int j) {
                wfield(i, j) = {4.0, -5.0, 6.0};
            });

        vector_field_type vvfield("vvfield", length, length);
        Kokkos::parallel_for(
            "assign", mdrange_t({0, 0}, {length, length}), KOKKOS_LAMBDA(const int i, const int j) {
                vvfield(i, j) =
                    0.25 * wfield(i, j) * cross(vfield(i, j) * wfield(i, j), wfield(i, j))
                    + wfield(i, j) / vfield(i, j) + 2.0;
            });

        Kokkos::fence();

        vector_field_type::HostMirror host_view = Kokkos::create_mirror_view(vvfield);
        Kokkos::deep_copy(host_view, vvfield);

        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                std::cout << host_view(i, j)[0] << " " << host_view(i, j)[1] << " "
                          << host_view(i, j)[2] << std::endl;
            }
            std::cout << std::endl;
        }

        typedef Kokkos::View<double**> scalar_field_type;
        scalar_field_type sfield("sfield", length, length);

        Kokkos::parallel_for(
            "assign", mdrange_t({0, 0}, {length, length}), KOKKOS_LAMBDA(const int i, const int j) {
                sfield(i, j) = dot(wfield(i, j), wfield(i, j)).apply();
            });

        Kokkos::fence();

        scalar_field_type::HostMirror host_sview = Kokkos::create_mirror_view(sfield);
        Kokkos::deep_copy(host_sview, sfield);

        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                std::cout << host_sview(i, j) << " ";
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
