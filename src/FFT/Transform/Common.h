#ifndef IPPL_COMMON_H
#define IPPL_COMMON_H

#include <array>

#include "Expression/IpplOperations.h"
#include "Utility/ParallelDispatch.h"
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
        constexpr unsigned Dim = InputViewT::rank;
        using index_array_type = typename ippl::RangePolicy<Dim, ExecSpace>::index_array_type;
        ippl::parallel_for(
            "FFT_toTemp", ippl::getRangePolicy(input, n_ghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                ippl::apply(output, args - n_ghost) = ippl::apply(input, args);
            });
    }

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
