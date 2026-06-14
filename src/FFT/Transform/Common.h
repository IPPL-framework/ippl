/*!
 * @file Common.h
 * @brief Helpers shared by the FFT transform wrappers.
 *
 * Provides domain<->box translation and ghost-cell-aware buffer copies
 * between BareFields and the LayoutLeft scratch buffers handed to the
 * underlying FFT backends.
 */
#ifndef IPPL_COMMON_H
#define IPPL_COMMON_H

#include <array>

#include "Expression/IpplOperations.h"
#include "Utility/ParallelDispatch.h"
#include "Utility/ViewUtils.h"

namespace ippl::fft {
    /*!
     * @brief Convert an IPPL NDIndex domain into low/high index triples.
     *
     * Pads unused dimensions (when @p Dim < 3) with 0 so the output is always
     * length 3 - the size that heFFTe / cuFFT(Mp) box descriptors expect.
     *
     * @tparam Dim     Active spatial dimension count.
     * @tparam NDIndex IPPL NDIndex type.
     * @param  domain Input NDIndex describing the local subdomain.
     * @param  low    Output: inclusive lower indices per axis.
     * @param  high   Output: inclusive upper indices per axis.
     */
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

    /*!
     * @brief Copy a Field view (with @p n_ghost halo) into a contiguous FFT buffer.
     *
     * The input view has ghost cells; the output is expected to be the
     * exact-size FFT buffer, so each index is shifted by @p n_ghost.
     *
     * @tparam ExecSpace   Kokkos execution space the launch is dispatched on.
     * @tparam OutputViewT Destination view type (no ghosts).
     * @tparam InputViewT  Source view type (with ghosts).
     */
    template <typename ExecSpace, typename OutputViewT, typename InputViewT>
    inline void copyToTemp(OutputViewT& output, const InputViewT& input, int n_ghost) {
        constexpr unsigned Dim = InputViewT::rank;
        using index_array_type = typename ippl::RangePolicy<Dim, ExecSpace>::index_array_type;
        ippl::parallel_for(
            "FFT_toTemp", ippl::getRangePolicy(input, n_ghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                ippl::apply(output, args - n_ghost) = ippl::apply(input, args);
            });
    }

    /*!
     * @brief Inverse of copyToTemp: scatter a contiguous FFT buffer back into a
     *        ghost-padded Field view.
     */
    template <typename ExecSpace, typename OutputViewT, typename InputViewT>
    inline void copyFromTemp(OutputViewT& output, const InputViewT& input, int n_ghost) {
        constexpr unsigned Dim = OutputViewT::rank;
        using index_array_type = typename ippl::RangePolicy<Dim, ExecSpace>::index_array_type;
        ippl::parallel_for(
            "FFT_fromTemp", ippl::getRangePolicy(output, n_ghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                ippl::apply(output, args) = ippl::apply(input, args - n_ghost);
            });
    }

}  // namespace ippl::fft

#endif  // IPPL_COMMON_H
