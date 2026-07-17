/*!
 * @file AtomicGather.h
 * @brief Per-particle gather kernel.
 *
 * Each particle reads its W^Dim stencil from the field. The "Atomic" name is
 * kept for symmetry with the scatter side; gather is read-only and never
 * actually issues atomic operations. With @c UseSorting=true a permutation
 * is applied so particles in the same tile read contiguous memory.
 */
#ifndef IPPL_ATOMIC_GATHER_H
#define IPPL_ATOMIC_GATHER_H

#include <Kokkos_Core.hpp>

#include "Interpolation/CoordinateTransform.h"
#include "Interpolation/Gather/GatherArgumentsBase.h"
#include "Interpolation/WidthDispatcher.h"

namespace ippl::Interpolation::detail {
    /*!
     * @struct AtomicGather
     * @brief Compile-time-width gather functor.
     * @tparam W          Compile-time kernel width.
     * @tparam Types      GatherTypes bundle.
     * @tparam UseSorting When true, particles are pre-binned for better locality.
     */
    template <int W, class Types, bool UseSorting = false>
    struct AtomicGather {
        static constexpr bool requires_binning = UseSorting;
        static constexpr unsigned Dim          = Types::Dim;

        using RealType        = typename Types::RealType;
        using ValueType       = typename Types::ValueType;
        using memory_space    = typename Types::memory_space;
        using execution_space = typename Types::execution_space;

        struct Arguments : GatherArgumentsBase<Arguments, Types> {
            using PermuteView = Kokkos::View<ippl::detail::size_type*, memory_space>;
            PermuteView permute;  // Only used when UseSorting = true

            template <class Field, class Positions, class Values, class Kernel>
            static Arguments create(const Field& field, const Positions& pos, Values& vals,
                                    const Kernel& k, const GatherConfig<Dim>& cfg,
                                    const GatherBinningResult<memory_space>& binning = {}) {
                Arguments a;
                a.initBase(field, pos, vals, k, cfg.add_to_attribute);
                if constexpr (UseSorting) {
                    a.permute = binning.permute;
                }
                return a;
            }
        };
        Arguments args;

        struct Stencil {
            Kokkos::Array<int, Dim> base;  // Stencil leftmost indices in all dims
            Kokkos::Array<Kokkos::Array<RealType, W>, Dim> kw;  // Precomputed kernel evals
        };

        KOKKOS_INLINE_FUNCTION void operator()(size_t j) const {
            using result_type = decltype(args.grid)::non_const_value_type;

            // Get actual particle index (sorted or direct)
            const size_t p = UseSorting ? args.permute(j) : j;

            // Build stencil
            CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};
            Stencil stencil{};
            for_constexpr(std::make_integer_sequence<int, Dim>{}, [&]<int d>() {
                const RealType g_pos    = transform.toGridCoordinate(args.x(p)[d], d);
                const RealType g_pos_cc = g_pos - RealType(0.5);
                const int idx0          = transform.getStencilBase(g_pos_cc, W);

                stencil.base[d] = idx0 - args.local_offset[d] + args.nghost;

                auto& kernel_vals = stencil.kw[d];
                for (int i = 0; i < W; ++i) {
                    kernel_vals[i] =
                        args.kernel((g_pos - (RealType(idx0 + i) + RealType(0.5))) * args.inv_hw);
                }
            });

            // Gather W^d stencil around non-uniform pt
            result_type out = result_type(0);
            auto rec        = [&]<unsigned D>(auto&& self, RealType wprod, auto... idx) -> void {
                const int bD   = get<D>(stencil.base);
                const auto& kD = get<D>(stencil.kw);

                for (int i = 0; i < W; ++i) {
                    const RealType w = wprod * kD[i];
                    if constexpr (D == 0) {
                        out += args.grid(bD + i, idx...) * w;
                    } else {
                        self.template operator()<D - 1>(self, w, bD + i, idx...);
                    }
                }
            };
            rec.template operator()<Dim - 1>(rec, RealType(1));

            if (args.add_to_attribute) {
                if constexpr (std::is_same_v<Kokkos::complex<RealType>, result_type>
                              && std::is_same_v<ValueType, RealType>) {
                    args.values(p) = args.values(p) + out.real();
                } else {
                    args.values(p) = args.values(p) + out;
                }
            } else {
                if constexpr (std::is_same_v<Kokkos::complex<RealType>, result_type>
                              && std::is_same_v<ValueType, RealType>) {
                    args.values(p) = out.real();
                } else {
                    args.values(p) = out;
                }
            }
        }

        void run(size_t n_particles) {
            auto policy = Kokkos::RangePolicy<execution_space>(0, n_particles);
            // `Kokkos::Experimental::prefer` + DesiredOccupancy is still in
            // the Experimental namespace (Kokkos 5.x). The interface may
            // move; if it does, drop the prefer() and pass `policy` directly.
            auto const policy_tuned = Kokkos::Experimental::prefer(
                policy, Kokkos::Experimental::DesiredOccupancy{Kokkos::AUTO});
            Kokkos::parallel_for("AtomicGather", policy_tuned, *this);
        }
    };
}  // namespace ippl::Interpolation::detail

#endif  // IPPL_ATOMIC_GATHER_H
