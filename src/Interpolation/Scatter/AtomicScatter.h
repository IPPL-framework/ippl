#ifndef IPPL_TEAM_ATOMIC_SCATTER_H
#define IPPL_TEAM_ATOMIC_SCATTER_H

#include <Kokkos_Core.hpp>

#include "Interpolation/CoordinateTransform.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"

namespace ippl::Interpolation::detail {
    template <typename RealType, int Dim, int W>
    struct KwPad {
        static constexpr int base   = Dim * W;
        static constexpr int pad    = (base & 1) ? 0 : 1;  // force stride odd
        static constexpr int stride = base + pad;
    };

    template <int W, class Types, class Policy>
    struct AtomicScatter {
        static constexpr bool requires_binning = false;
        static constexpr unsigned Dim          = Types::Dim;

        using RealType        = typename Types::RealType;
        using ValueType       = typename Types::ValueType;
        using memory_space    = typename Types::memory_space;
        using execution_space = typename Types::execution_space;

        using team_policy = Kokkos::TeamPolicy<execution_space, Kokkos::LaunchBounds<64, 16>>;
        using team_member = typename team_policy::member_type;

        using scratch_space = typename execution_space::scratch_memory_space;
        using unmanaged     = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

        // Total number of stencil points = W^Dim
        static constexpr int total_stencil_points = []() {
            int result = 1;
            for (unsigned d = 0; d < Dim; ++d)
                result *= W;
            return result;
        }();

        // Warp size (vector length)
        static constexpr int vector_length = 32;

        // Scratch view types for N particles
        using ScratchBaseView    = Kokkos::View<int**, scratch_space, unmanaged>;
        using ScratchWeightsView = Kokkos::View<RealType***, scratch_space, unmanaged>;
        using ScratchValuesView  = Kokkos::View<ValueType*, scratch_space, unmanaged>;
        using G0View             = Kokkos::View<RealType**, scratch_space, unmanaged>;

        struct Arguments : ScatterArgumentsBase<Arguments, Types> {
            using PermuteView = Kokkos::View<uint64_t*, memory_space>;
            PermuteView permute;

            template <class Field, class Positions, class Values, class Kernel>
            static Arguments create(Field& field, const Positions& pos, const Values& vals,
                                    const Kernel& k, const ScatterConfig<Dim>&,
                                    const BinningResult<Dim, memory_space>& binning = {}) {
                Arguments a;
                a.initBase(field, pos, vals, k);
                if constexpr (Policy::use_sorting) {
                    a.permute = binning.permute;
                }
                return a;
            }
        };

        // Compute scratch size for N particles per team
        template <bool>
        static size_t compute_scratch_size(int particles_per_team) {
            return ScratchBaseView::shmem_size(particles_per_team, Dim)
                   + ScratchWeightsView::shmem_size(particles_per_team, Dim, W)
                   + G0View::shmem_size(particles_per_team, Dim);
        }

        template <bool>
        static size_t compute_scratch_size(Vector<int, 3> /* tile_size */, int /* team_size */,
                                           int /* z_batches */ = 1) {
            return 0;
        }

        Arguments args;
        int particles_per_team_;

        AtomicScatter(const Arguments& a)
            : args(a)
            , particles_per_team_(2) {}

        // Convert linear stencil index to multi-dimensional indices
        KOKKOS_INLINE_FUNCTION Kokkos::Array<int, Dim> linear_to_multi(int linear_idx) const {
            Kokkos::Array<int, Dim> idx;
            for (unsigned d = 0; d < Dim; ++d) {
                idx[d] = linear_idx % W;
                linear_idx /= W;
            }
            return idx;
        }

        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t = typename decltype(args.grid)::non_const_value_type;

            const int team_rank = team.team_rank();
            const int team_size = team.team_size();

            const size_t base_particle = size_t(team.league_rank()) * size_t(team_size);
            size_t p_global            = base_particle + size_t(team_rank);
            if (p_global >= args.n_particles)
                return;

            if constexpr (Policy::use_sorting)
                p_global = args.permute(p_global);

            CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};

            // -------------------------
            // scratch weights: 1D + computed padding (force odd stride for W=2..10)
            // -------------------------
            using ScratchSpace = typename team_member::scratch_memory_space;
            using Unmanaged    = Kokkos::MemoryUnmanaged;

            auto scratch = team.team_scratch(0);

            constexpr int KW_BASE   = int(Dim) * int(W);
            constexpr int KW_PAD    = (KW_BASE & 1) ? 0 : 1;  // make stride odd
            constexpr int KW_STRIDE = KW_BASE + KW_PAD;

            using Kw1D = Kokkos::View<RealType*, ScratchSpace, Unmanaged>;
            Kw1D kw(scratch, team_size * KW_STRIDE);

            RealType* kw_ptr = kw.data() + team_rank * KW_STRIDE;

            const RealType inv_hw = args.inv_hw;

            // helper: fill weights for a compile-time dimension D using g0 computed on-demand
            auto fill_dim = [&](auto Dtag, RealType g0d) {
                constexpr int D = decltype(Dtag)::value;
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W), [&](const int i) {
                    const RealType xi = g0d - RealType(i) * inv_hw;
                    if constexpr (Types::KernelType::has_width_template)
                        kw_ptr[D * W + i] = args.kernel.template eval<W>(xi);
                    else
                        kw_ptr[D * W + i] = args.kernel(xi);
                });
            };

            // -------------------------
            // Dimension-specialized paths (LayoutLeft only)
            // Compute (base, g0) only where they’re needed and keep live ranges tight.
            // -------------------------
            if constexpr (Dim == 1) {
                // compute dim0 and immediately fill weights
                const RealType x0  = args.x(p_global)[0];
                const RealType gp0 = transform.template toGridCoordinate<0>(x0);
                const int idx0     = transform.template getStencilBase<W>(gp0 - RealType(0.5));
                const int b0       = idx0 - args.local_offset[0] + args.nghost;
                fill_dim(std::integral_constant<int, 0>{},
                         (gp0 - (RealType(idx0) + RealType(0.5))) * inv_hw);

                // stencil
                const ValueType my_val = args.values(p_global);
                auto grid              = args.grid;

                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, W), [&](const int i0) {
                    const RealType w = kw_ptr[0 * W + i0];
                    auto& cell       = grid(b0 + i0);
                    Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
                                       static_cast<grid_value_t>(my_val * w));
                });

            } else if constexpr (Dim == 2) {
                // dim0
                const RealType x0  = args.x(p_global)[0];
                const RealType gp0 = transform.template toGridCoordinate<0>(x0);
                const int idx0     = transform.template getStencilBase<W>(gp0 - RealType(0.5));
                const int b0       = idx0 - args.local_offset[0] + args.nghost;
                fill_dim(std::integral_constant<int, 0>{},
                         (gp0 - (RealType(idx0) + RealType(0.5))) * inv_hw);

                // dim1
                const RealType x1  = args.x(p_global)[1];
                const RealType gp1 = transform.template toGridCoordinate<1>(x1);
                const int idx1     = transform.template getStencilBase<W>(gp1 - RealType(0.5));
                const int b1       = idx1 - args.local_offset[1] + args.nghost;
                fill_dim(std::integral_constant<int, 1>{},
                         (gp1 - (RealType(idx1) + RealType(0.5))) * inv_hw);

                // stencil: LayoutLeft => i0 fastest, then i1
                const ValueType my_val     = args.values(p_global);
                auto grid                  = args.grid;
                constexpr int STENCIL_SIZE = W * W;

                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, STENCIL_SIZE),
                                     [&](const int flat) {
                                         const int i0 = flat % W;
                                         const int i1 = flat / W;

                                         const RealType w = kw_ptr[0 * W + i0] * kw_ptr[1 * W + i1];
                                         auto& cell       = grid(b0 + i0, b1 + i1);
                                         Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
                                                            static_cast<grid_value_t>(my_val * w));
                                     });

            } else if constexpr (Dim == 3) {
                // dim0
                const RealType x0  = args.x(p_global)[0];
                const RealType gp0 = transform.template toGridCoordinate<0>(x0);
                const int idx0     = transform.template getStencilBase<W>(gp0 - RealType(0.5));
                const int b0       = idx0 - args.local_offset[0] + args.nghost;
                fill_dim(std::integral_constant<int, 0>{},
                         (gp0 - (RealType(idx0) + RealType(0.5))) * inv_hw);

                // dim1
                const RealType x1  = args.x(p_global)[1];
                const RealType gp1 = transform.template toGridCoordinate<1>(x1);
                const int idx1     = transform.template getStencilBase<W>(gp1 - RealType(0.5));
                const int b1       = idx1 - args.local_offset[1] + args.nghost;
                fill_dim(std::integral_constant<int, 1>{},
                         (gp1 - (RealType(idx1) + RealType(0.5))) * inv_hw);

                // dim2
                const RealType x2  = args.x(p_global)[2];
                const RealType gp2 = transform.template toGridCoordinate<2>(x2);
                const int idx2     = transform.template getStencilBase<W>(gp2 - RealType(0.5));
                const int b2       = idx2 - args.local_offset[2] + args.nghost;
                fill_dim(std::integral_constant<int, 2>{},
                         (gp2 - (RealType(idx2) + RealType(0.5))) * inv_hw);

                // stencil: LayoutLeft => i0 fastest, then i1, then i2
                const ValueType my_val     = args.values(p_global);
                auto grid                  = args.grid;
                constexpr int STENCIL_SIZE = W * W * W;

                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team, STENCIL_SIZE), [&](const int flat) {
                        const int i0 = flat % W;
                        const int t  = flat / W;
                        const int i1 = t % W;
                        const int i2 = t / W;

                        const RealType w =
                            kw_ptr[0 * W + i0] * kw_ptr[1 * W + i1] * kw_ptr[2 * W + i2];

                        auto& cell = grid(b0 + i0, b1 + i1, b2 + i2);
                        Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
                                           static_cast<grid_value_t>(my_val * w));
                    });
            }
        }

        void run(size_t) {
            const int team_size       = particles_per_team_;
            const size_t scratch_size = compute_scratch_size<true>(team_size);
            const size_t n_teams      = (args.n_particles + team_size - 1) / team_size;

            team_policy policy(n_teams, team_size, vector_length);
            policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

            Kokkos::parallel_for("AtomicScatterVectorized", policy, *this);
        }
    };

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_TEAM_ATOMIC_SCATTER_H