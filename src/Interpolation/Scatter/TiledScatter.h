// Interpolation/Scatter/TiledScatter.h
#ifndef IPPL_TILED_SCATTER_H
#define IPPL_TILED_SCATTER_H

#include <Kokkos_Core.hpp>

#include "Interpolation/CoordinateTransform.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"

namespace ippl::Interpolation::detail {

    template <int W, class Types, class Policy>
    struct TiledScatter {
        static_assert(Policy::use_sorting,
                      "TiledScatter assumes sorted/bin-partitioned particles (Policy::use_sorting "
                      "must be true).");

        static constexpr bool requires_binning = true;  // algorithmically required
        static constexpr unsigned Dim          = Types::Dim;
        static constexpr int half_left         = (W + 1) / 2;

        using RealType        = typename Types::RealType;
        using ValueType       = typename Types::ValueType;
        using memory_space    = typename Types::memory_space;
        using execution_space = typename Types::execution_space;

        using team_policy   = Kokkos::TeamPolicy<execution_space>;
        using team_member   = typename team_policy::member_type;
        using scratch_space = typename execution_space::scratch_memory_space;

        using scratch_view =
            Kokkos::View<RealType*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        struct Arguments : ScatterArgumentsBase<Arguments, Types> {
            Kokkos::View<uint64_t*, memory_space> permute;
            Kokkos::View<uint64_t*, memory_space> bin_offsets;
            Vector<int, Dim> num_tiles;
            Vector<int, Dim> tile_size;
            int team_size;

            template <class Field, class Positions, class Values, class Kernel>
            static Arguments create(Field& field, const Positions& pos, const Values& vals,
                                    const Kernel& k, const ScatterConfig<Dim>& config,
                                    const BinningResult<Dim, memory_space>& binning = {}) {
                Arguments a;
                a.initBase(field, pos, vals, k);
                a.permute     = binning.permute;
                a.bin_offsets = binning.bin_offsets;
                a.num_tiles   = binning.num_tiles;
                a.tile_size   = config.get_tile_size();
                // Defensive backend clamp: a CSV preset tagged for sm_xx may
                // be loaded on a build where the test fixture also instantiates
                // host execution spaces. Force team_size=1 there so we never
                // construct a TeamPolicy<Serial> with team_size > 1, which
                // aborts with "Requested Team Size is too large!". Scatter.h
                // already routes Serial through Atomic, so this branch is
                // belt-and-braces for any future host-only backend.
#ifdef KOKKOS_ENABLE_SERIAL
                constexpr bool host_only =
                    std::is_same_v<execution_space, Kokkos::Serial>;
#else
                constexpr bool host_only = false;
#endif
                a.team_size = host_only ? 1 : config.team_size;
                return a;
            }
        };

        Arguments args;

        struct Stencil {
            Kokkos::Array<int, Dim> base;
            Kokkos::Array<Kokkos::Array<RealType, W>, Dim> kw;
        };

        template <typename GridValueT>
        struct Histogram {
            static constexpr bool is_complex =
                std::is_same_v<GridValueT, Kokkos::complex<RealType>>;

            scratch_view data_r;
            scratch_view data_i;  // only used if is_complex
            Vector<int, Dim> size;
            size_t total;

            KOKKOS_INLINE_FUNCTION void init(const team_member& team, scratch_view r,
                                             scratch_view i, Vector<int, Dim> sz, size_t n) {
                data_r = r;
                if constexpr (is_complex)
                    data_i = i;
                size  = sz;
                total = n;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total), [&](size_t idx) {
                    data_r(idx) = 0;
                    if constexpr (is_complex)
                        data_i(idx) = 0;
                });
            }

            KOKKOS_INLINE_FUNCTION size_t to_flat(const Kokkos::Array<int, Dim>& c) const {
                size_t idx = 0, stride = 1;
                for (unsigned d = 0; d < Dim; ++d) {
                    idx += static_cast<size_t>(c[d]) * stride;
                    stride *= static_cast<size_t>(size[d]);
                }
                return idx;
            }

            KOKKOS_INLINE_FUNCTION Kokkos::Array<int, Dim> from_flat(size_t idx) const {
                Kokkos::Array<int, Dim> c{};
                for (unsigned d = 0; d < Dim; ++d) {
                    c[d] = static_cast<int>(idx % static_cast<size_t>(size[d]));
                    idx /= static_cast<size_t>(size[d]);
                }
                return c;
            }

            KOKKOS_INLINE_FUNCTION bool in_bounds(const Kokkos::Array<int, Dim>& c) const {
                for (unsigned d = 0; d < Dim; ++d) {
                    if (c[d] < 0 || c[d] >= size[d])
                        return false;
                }
                return true;
            }

            template <typename V>
            KOKKOS_INLINE_FUNCTION void atomic_add(const Kokkos::Array<int, Dim>& c, const V& v,
                                                   RealType w) {
                if (!in_bounds(c))
                    return;
                const size_t idx = to_flat(c);

                if constexpr (is_complex && std::is_same_v<V, Kokkos::complex<RealType>>) {
                    Kokkos::atomic_add(&data_r(idx), v.real() * w);
                    Kokkos::atomic_add(&data_i(idx), v.imag() * w);
                } else if constexpr (is_complex) {
                    Kokkos::atomic_add(&data_r(idx), v * w);
                } else {
                    Kokkos::atomic_add(&data_r(idx), v * w);
                }
            }

            KOKKOS_INLINE_FUNCTION GridValueT operator()(size_t idx) const {
                if constexpr (is_complex)
                    return GridValueT(data_r(idx), data_i(idx));
                else
                    return data_r(idx);
            }
        };

        KOKKOS_INLINE_FUNCTION Vector<int, Dim> hist_size() const {
            Vector<int, Dim> hs;
            for (unsigned d = 0; d < Dim; ++d)
                hs[d] = args.tile_size[d] + W + 1;
            return hs;
        }

        KOKKOS_INLINE_FUNCTION size_t hist_total() const {
            size_t n = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n *= static_cast<size_t>(args.tile_size[d] + W + 1);
            return n;
        }

        KOKKOS_INLINE_FUNCTION Vector<int, Dim> decode_tile_base(size_t tile_id) const {
            Vector<int, Dim> tile_base;
            for (size_t t = tile_id, d = Dim; d-- > 0;) {
                tile_base[d] = static_cast<int>(t % static_cast<size_t>(args.num_tiles[d]))
                               * args.tile_size[d];
                t /= static_cast<size_t>(args.num_tiles[d]);
            }
            return tile_base;
        }

        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t = typename decltype(args.grid)::non_const_value_type;

            const size_t tile_id = team.league_rank();
            const auto tile_base = decode_tile_base(tile_id);

            const auto hs      = hist_size();
            const size_t total = hist_total();

            // Per-team scratch allocation: bump allocator semantics (your point).
            scratch_view scratch_r(team.team_scratch(0), total);
            scratch_view scratch_i;
            if constexpr (Histogram<grid_value_t>::is_complex) {
                scratch_i = scratch_view(team.team_scratch(0), total);
            }

            Histogram<grid_value_t> hist;
            hist.init(team, scratch_r, scratch_i, hs, total);
            team.team_barrier();

            // Hoist transform: same for all particles.
            const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
                                                               args.n_grid};

            const size_t pstart = args.bin_offsets(tile_id);
            const size_t pend   = args.bin_offsets(tile_id + 1);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, pstart, pend), [&](size_t ip) {
                const size_t p = args.permute(ip);
                const auto val = args.values(p);

                Stencil stencil{};

                for_constexpr(std::make_integer_sequence<int, Dim>{}, [&]<int d> {
                    const RealType g = transform.toGridCoordinate(args.x(p)[d], d);
                    const int idx0   = transform.getStencilBase(g - RealType(0.5), W);

                    stencil.base[d] = idx0 - args.local_offset[d];

                    for (int i = 0; i < W; ++i) {
                        stencil.kw[d][i] = args.kernel((g - (RealType(idx0 + i) + RealType(0.5))) * args.inv_hw);
                    }
                });

                auto scatter = [&]<unsigned D>(auto&& self, RealType wprod,
                                               Kokkos::Array<int, Dim> hc) {
                    for (int i = 0; i < W; ++i) {
                        hc[D]            = stencil.base[D] + i + half_left - tile_base[D];
                        const RealType w = wprod * stencil.kw[D][i];

                        if constexpr (D == 0) {
                            hist.atomic_add(hc, val, w);
                        } else {
                            self.template operator()<D - 1>(self, w, hc);
                        }
                    }
                };
                scatter.template operator()<Dim - 1>(scatter, RealType(1), {});
            });

            team.team_barrier();

            // Flush histogram to grid
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total), [&](size_t idx) {
                const auto hc = hist.from_flat(idx);

                Kokkos::Array<int, Dim> gc{};
                for (unsigned d = 0; d < Dim; ++d) {
                    const int local = tile_base[d] + hc[d] - half_left;
                    if (local < -args.nghost || local >= args.n_grid_local[d] + args.nghost)
                        return;
                    gc[d] = local + args.nghost;
                }

                [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    // There is some bug in Kokkos 5.0.0 that leads to deadlocks with complex
                    // atomics and that I was unable of reproducing in a MWE
                    if constexpr (Histogram<grid_value_t>::is_complex) {
                        RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
                        Kokkos::atomic_add(&ptr[0], hist.data_r(idx));
                        Kokkos::atomic_add(&ptr[1], hist.data_i(idx));
                    } else {
                        Kokkos::atomic_add(&args.grid(gc[Is]...),
                                           static_cast<grid_value_t>(hist(idx)));
                    }
                }(std::make_index_sequence<Dim>{});
            });
        }

        template <bool IsComplex>
        static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int /* team_size */,
                                           int /* z_batches */ = 1) {
            size_t n = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n *= static_cast<size_t>(tile_size[d] + W + 1);
            // Use shmem_size() to account for Kokkos alignment overhead
            return (IsComplex ? 2 : 1) * scratch_view::shmem_size(n);
        }

        void run(size_t) {
            using grid_value_t   = typename decltype(args.grid)::non_const_value_type;
            constexpr bool cplx  = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            const size_t scratch = compute_scratch_size<cplx>(args.tile_size, 0);

            size_t n_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n_tiles *= static_cast<size_t>(args.num_tiles[d]);

            Kokkos::parallel_for(
                "TiledScatter",
                team_policy(n_tiles, args.team_size).set_scratch_size(0, Kokkos::PerTeam(scratch)),
                *this);
        }
    };

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_TILED_SCATTER_H
