#ifndef IPPL_COMMON_H
#define IPPL_COMMON_H

#include <array>

#include "Utility/ViewUtils.h"

namespace ippl::fft {
    template <unsigned Dim, typename NDIndex>
    inline void domainToBounds(const NDIndex& domain, std::array<long long, 3>& low,
                               std::array<long long, 3>& high) {
        low.fill(0);
        high.fill(0);
        for (unsigned d = 0; d < Dim; ++d) {
            low[d]  = static_cast<long long>(domain[d].first());
            high[d] = static_cast<long long>(domain[d].first() + domain[d].length() - 1);
        }
    }

    template <typename ExecSpace, typename OutputViewT, typename InputViewT>
    inline void copyToTemp(OutputViewT& output, const InputViewT& input, int n_ghost) {
        Kokkos::parallel_for(
            "FFT_CC_toTemp",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                {n_ghost, n_ghost, n_ghost}, {static_cast<int>(input.extent(0)) - n_ghost,
                                              static_cast<int>(input.extent(1)) - n_ghost,
                                              static_cast<int>(input.extent(2)) - n_ghost}),
            KOKKOS_LAMBDA(int i, int j, int k) {
                output(i - n_ghost, j - n_ghost, k - n_ghost) = input(i, j, k);
            });
    }
    template <typename ExecSpace, typename OutputViewT, typename InputViewT>
    inline void copyFromTemp(OutputViewT& output, const InputViewT& input, int n_ghost) {
        Kokkos::parallel_for(
            "FFT_CC_fromTemp",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>(
                {n_ghost, n_ghost, n_ghost}, {static_cast<int>(output.extent(0)) - n_ghost,
                                              static_cast<int>(output.extent(1)) - n_ghost,
                                              static_cast<int>(output.extent(2)) - n_ghost}),
            KOKKOS_LAMBDA(int i, int j, int k) {
                output(i, j, k) = input(i - n_ghost, j - n_ghost, k - n_ghost);
            });
    }

}  // namespace ippl::fft

#endif  // IPPL_COMMON_H
