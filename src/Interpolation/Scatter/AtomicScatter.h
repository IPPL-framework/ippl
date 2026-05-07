#ifndef IPPL_TEAM_ATOMIC_SCATTER_H
#define IPPL_TEAM_ATOMIC_SCATTER_H

#include <Kokkos_Core.hpp>

#include "Interpolation/CoordinateTransform.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"

namespace ippl::Interpolation::detail {
    template <int W, class Types, class Policy>
    struct AtomicScatter {
        static constexpr bool requires_binning = false;
        static constexpr unsigned Dim          = Types::Dim;

        using RealType        = typename Types::RealType;
        using ValueType       = typename Types::ValueType;
        using memory_space    = typename Types::memory_space;
        using execution_space = typename Types::execution_space;

        using team_policy = Kokkos::TeamPolicy<execution_space>;
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

        // Per-backend defaults. Host backends require vector_length=1 and
        // team_size=1; HIP wavefronts are 64-wide; CUDA warps are 32-wide.
        static constexpr bool is_serial =
#ifdef KOKKOS_ENABLE_SERIAL
            std::is_same_v<execution_space, Kokkos::Serial>;
#else
            false;
#endif
        static constexpr bool is_openmp =
#ifdef KOKKOS_ENABLE_OPENMP
            std::is_same_v<execution_space, Kokkos::OpenMP>;
#else
            false;
#endif
        static constexpr bool is_hip =
#ifdef KOKKOS_ENABLE_HIP
            std::is_same_v<execution_space, Kokkos::HIP>;
#else
            false;
#endif
        static constexpr bool is_host_backend = is_serial || is_openmp;
        static constexpr int  vector_length   = is_host_backend ? 1 : (is_hip ? 64 : 32);

        // Scratch view types for N particles
        using ScratchBaseView    = Kokkos::View<int**, scratch_space, unmanaged>;
        using ScratchWeightsView = Kokkos::View<RealType***, scratch_space, unmanaged>;
        using ScratchValuesView  = Kokkos::View<ValueType*, scratch_space, unmanaged>;
        using G0View             = Kokkos::View<RealType**, scratch_space, unmanaged>;

        struct Arguments : ScatterArgumentsBase<Arguments, Types> {
            using PermuteView = Kokkos::View<ippl::detail::size_type*, memory_space>;
            PermuteView permute;
            int particles_per_team = 1;

            template <class Field, class Positions, class Values, class Kernel>
            static Arguments create(Field& field, const Positions& pos, const Values& vals,
                                    const Kernel& k, const ScatterConfig<Dim>& cfg,
                                    const BinningResult<Dim, memory_space>& binning = {}) {
                Arguments a;
                a.initBase(field, pos, vals, k);
                a.particles_per_team =
                    is_serial ? 1 : (cfg.team_size > 0 ? cfg.team_size : 2);
                if constexpr (Policy::use_sorting) {
                    a.permute = binning.permute;
                }
                return a;
            }
        };

        // Compute per-team scratch size for `particles_per_team` particles.
        template <bool /*IsComplex*/>
        static size_t compute_scratch_size(int particles_per_team) {
            return ScratchBaseView::shmem_size(particles_per_team, Dim)
                   + ScratchWeightsView::shmem_size(particles_per_team, Dim, W)
                   + G0View::shmem_size(particles_per_team, Dim);
        }

        // Uniform interface required by Scatter::clamp_tile_to_shmem; AtomicScatter
        // has nothing to clamp (requires_binning == false), so this returns 0 and
        // is reached only via the discarded `if constexpr` branch.
        template <bool /*IsComplex*/>
        static size_t compute_scratch_size(const Vector<int, Dim>& /*tile_size*/,
                                           int /*team_size*/, int /*z_batches*/ = 1) {
            return 0;
        }

        Arguments args;

        AtomicScatter(const Arguments& a)
            : args(a) {}

        // Convert linear stencil index to multi-dimensional indices
        KOKKOS_INLINE_FUNCTION Kokkos::Array<int, Dim> linear_to_multi(int linear_idx) const {
            Kokkos::Array<int, Dim> idx;
            for (unsigned d = 0; d < Dim; ++d) {
                idx[d] = linear_idx % W;
                linear_idx /= W;
            }
            return idx;
        }

        // ------------------------------------------------------------------
        // Simple flat path used for narrow kernels (W < 3, e.g. NGP / CIC).
        // ------------------------------------------------------------------
        KOKKOS_INLINE_FUNCTION void operator()(const size_t i_in) const {
            using grid_value_t = typename decltype(args.grid)::non_const_value_type;

            size_t p = i_in;
            if constexpr (Policy::use_sorting)
                p = args.permute(i_in);

            CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx, args.n_grid};
            const RealType inv_hw = args.inv_hw;
            const ValueType v     = args.values(p);

            // Per-dimension stencil base + per-stencil-point weights.
            int      base[Dim];
            RealType kw[Dim][W];
            for (unsigned d = 0; d < Dim; ++d) {
                const RealType g     = transform.toGridCoordinate(args.x(p)[d], d);
                const int      idx   = transform.getStencilBase(g - RealType(0.5), W);
                base[d]              = idx - args.local_offset[d] + args.nghost;
                const RealType g0    = (g - (RealType(idx) + RealType(0.5))) * inv_hw;
                for (int k = 0; k < W; ++k) {
                    const RealType xi = g0 - RealType(k) * inv_hw;
                    if constexpr (Types::KernelType::has_width_template) {
                        kw[d][k] = args.kernel.template eval<W>(xi);
                    } else {
                        kw[d][k] = args.kernel(xi);
                    }
                }
            }

            auto grid = args.grid;
            if constexpr (Dim == 1) {
                for (int i0 = 0; i0 < W; ++i0) {
                    auto& cell = grid(base[0] + i0);
                    Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
                                       static_cast<grid_value_t>(v * kw[0][i0]));
                }
            } else if constexpr (Dim == 2) {
                for (int i1 = 0; i1 < W; ++i1) {
                    for (int i0 = 0; i0 < W; ++i0) {
                        auto& cell       = grid(base[0] + i0, base[1] + i1);
                        const RealType w = kw[0][i0] * kw[1][i1];
                        Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
                                           static_cast<grid_value_t>(v * w));
                    }
                }
            } else if constexpr (Dim == 3) {
                for (int i2 = 0; i2 < W; ++i2) {
                    for (int i1 = 0; i1 < W; ++i1) {
                        for (int i0 = 0; i0 < W; ++i0) {
                            auto& cell = grid(base[0] + i0, base[1] + i1, base[2] + i2);
                            const RealType w = kw[0][i0] * kw[1][i1] * kw[2][i2];
                            Kokkos::atomic_add(&to_grid_value<grid_value_t>(cell),
                                               static_cast<grid_value_t>(v * w));
                        }
                    }
                }
            }
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
            if constexpr (W < 3) {
                // NGP / CIC: stencils are too small for team-policy overhead
                // to pay back. Flat parallel_for with global atomic_add wins
                // (matches the original IPPL atomic-scatter pattern).
                Kokkos::parallel_for(
                    "AtomicScatter::Simple",
                    Kokkos::RangePolicy<execution_space>(0, args.n_particles), *this);
                return;
            }

            // Team-based path for W >= 3.
            // GPU per-team thread limit: team_size * vector_length must fit in
            // 1024 threads (HIP enforces strictly less-than 1024). Defensive
            // clamp here protects against CSV/auto-tune values produced for a
            // different (e.g. CUDA) device being replayed on HIP.
            int team_size = args.particles_per_team;
            if constexpr (!is_host_backend) {
                const int hw_cap = is_hip ? 1023 : 1024;
                const int max_team_size = std::max(1, hw_cap / vector_length);
                if (team_size > max_team_size) team_size = max_team_size;
            }
            if (team_size < 1) team_size = 1;

            const size_t scratch_size = compute_scratch_size<true>(team_size);
            const size_t n_teams      = (args.n_particles + team_size - 1) / team_size;

            team_policy policy(n_teams, team_size, vector_length);
            policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

            Kokkos::parallel_for("AtomicScatter", policy, *this);
        }
    };

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_TEAM_ATOMIC_SCATTER_H
