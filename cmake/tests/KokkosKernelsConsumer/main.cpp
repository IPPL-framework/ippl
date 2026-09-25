#include <Kokkos_Core.hpp>

#include <KokkosBlas3_gemm.hpp>
#include <KokkosKernels_config.h>
#if defined(KOKKOSKERNELS_ENABLE_TPL_MKL)
#include <mkl.h>
#elif defined(KOKKOSKERNELS_ENABLE_TPL_LAPACKE)
#include <lapacke.h>
#else
#error "This consumer test requires a host eigenanalysis provider"
#endif
#include <cmath>
int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    Kokkos::View<double**> a("a", 2, 2), b("b", 2, 2), c("c", 2, 2);
    Kokkos::deep_copy(a, 1.0);
    Kokkos::deep_copy(b, 2.0);
    KokkosBlas::gemm("N", "N", 1.0, a, b, 0.0, c);
    auto h      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
    double m[4] = {0, -1, 1, 0}, er[2], ei[2], l[4], r[4];
    auto info   = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'V', 'V', 2, m, 2, er, ei, l, 2, r, 2);
    return info || !std::isfinite(h(0, 0)) || !std::isfinite(ei[0]) || std::abs(h(0, 0) - 4) > 1e-14
           || std::abs(std::abs(ei[0]) - 1) > 1e-12;
}
