/*!
 * @file GridParallelScatter.h
 * @brief Output-tile-parallel scatter (one team per output tile).
 *
 * Particles must be pre-binned by output tile. Each Kokkos team owns one
 * output tile, walks its bin and accumulates into team scratch, then
 * flushes to global memory once. Lower atomics traffic than AtomicScatter
 * at high ppc; higher launch cost than TiledScatter for tiny tiles.
 */
#ifndef IPPL_GRID_PARALLEL_SCATTER_H
#define IPPL_GRID_PARALLEL_SCATTER_H

#include <type_traits>

namespace ippl::Interpolation::detail {

    //! Compile-time integer power: @c int_pow(base, exp) == @c base^exp.
    constexpr int int_pow(int base, int exp) {
        int result = 1;
        for (int i = 0; i < exp; ++i)
            result *= base;
        return result;
    }

    /*!
     * @struct GridParallelScatterTuning
     * @brief SFINAE-based detector for an optional Policy::fixed_oversubscription
     *        flag; defaults to true when the policy doesn't expose one.
     */
    template <class Policy, class = void>
    struct GridParallelScatterTuning {
        static constexpr bool fixed_oversubscription = true;
    };
    template <class Policy>
    struct GridParallelScatterTuning<Policy,
                                     std::void_t<decltype(Policy::fixed_oversubscription)>> {
        static constexpr bool fixed_oversubscription = Policy::fixed_oversubscription;
    };

    // -----------------------------------------------------------------------------
    // 3D optimized implementation
    // -----------------------------------------------------------------------------

    /*!
     * @struct GridParallelScatterImpl3D
     * @brief 3D-only output-tile-parallel scatter functor.
     * @tparam W      Compile-time kernel width.
     * @tparam Types  ScatterTypes bundle.
     * @tparam Policy Sort policy (must enable @c use_sorting).
     */
    template <int W, class Types, class Policy>
    struct GridParallelScatterImpl3D {
        static_assert(Policy::use_sorting,
                      "GridParallelScatter assumes sorted/bin-partitioned particles");
        static_assert(Types::Dim == 3, "GridParallelScatterImpl3D requires Dim == 3");

        static constexpr bool requires_binning = true;
        static constexpr unsigned Dim          = Types::Dim;
        static constexpr int half_left         = (W + 1) / 2;
        static constexpr int padded_extra      = 2 * half_left;
        static constexpr int stencil_total     = W * W * W;
        static constexpr bool fixed_oversubscription =
            GridParallelScatterTuning<Policy>::fixed_oversubscription;

        // Kernel weight stride per particle: Dim*W.
        // No extra padding - modest store conflicts on kerevals fill are cheaper
        // than the shmem they would consume (which directly reduces block occupancy).
        static constexpr int ker_stride = 3 * W;

        using RealType        = typename Types::RealType;
        using ValueType       = typename Types::ValueType;
        using memory_space    = typename Types::memory_space;
        using execution_space = typename Types::execution_space;

        using team_policy   = Kokkos::TeamPolicy<execution_space>;
        using team_member   = typename team_policy::member_type;
        using scratch_space = typename execution_space::scratch_memory_space;

        using scratch_real_view =
            Kokkos::View<RealType*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using scratch_int_view =
            Kokkos::View<int*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        struct Arguments : ScatterArgumentsBase<Arguments, Types> {
            Kokkos::View<size_t*, memory_space> permute;
            Kokkos::View<size_t*, memory_space> bin_offsets;
            Vector<int, Dim> num_tiles;
            Vector<int, Dim> tile_size;
            int team_size;
            int oversubscription_factor;
            int batch_np;

            template <class Field, class Positions, class Values, class Kernel>
            static Arguments create(Field& field, const Positions& pos, const Values& vals,
                                    const Kernel& k, const ScatterConfig<Dim>& config,
                                    const BinningResult<Dim, memory_space>& binning) {
                Arguments a;
                a.initBase(field, pos, vals, k);
                a.permute                 = binning.permute;
                a.bin_offsets             = binning.bin_offsets;
                a.num_tiles               = binning.num_tiles;
                a.tile_size               = config.get_tile_size();
                // Defensive: clamp team_size on host-only backends (see
                // TiledScatter Arguments::create for rationale).
#ifdef KOKKOS_ENABLE_SERIAL
                constexpr bool host_only =
                    std::is_same_v<execution_space, Kokkos::Serial>;
#else
                constexpr bool host_only = false;
#endif
                a.team_size               = host_only ? 1 : config.team_size;
                a.oversubscription_factor = config.oversubscription_factor;
                a.batch_np                = config.z_batches > 0 ? config.z_batches : 1;
                return a;
            }
        };

        Arguments args;
        size_t total_              = 1;  // hs0 * hs1 * hs2, no pitch padding
        size_t sub_teams_per_tile_ = 1;

        // binning flattens with dimension 2 fastest:
        //   bin = tx*(ny*nz) + ty*nz + tz -> decode in order z, y, x.
        KOKKOS_INLINE_FUNCTION void decode_tile_base(const size_t tile_id_in, int& tx, int& ty,
                                                     int& tz) const {
            size_t t         = tile_id_in;
            const size_t nt2 = static_cast<size_t>(args.num_tiles[2]);
            const size_t nt1 = static_cast<size_t>(args.num_tiles[1]);
            const int iz     = static_cast<int>(t % nt2);
            t /= nt2;
            const int iy = static_cast<int>(t % nt1);
            t /= nt1;
            const int ix = static_cast<int>(t);
            tx           = ix * args.tile_size[0];
            ty           = iy * args.tile_size[1];
            tz           = iz * args.tile_size[2];
        }

        template <bool NeedsImag>
        KOKKOS_FORCEINLINE_FUNCTION static void scatter_particle_fast(
            const team_member& team, const int sx, const int sy, const int sz, const int pitch0,
            const int pitch1, const RealType* __restrict__ kw, const RealType vr, const RealType vi,
            RealType* __restrict__ local_r, RealType* __restrict__ local_i) {
            const RealType* __restrict__ kwx = kw;
            const RealType* __restrict__ kwy = kw + W;
            const RealType* __restrict__ kwz = kw + 2 * W;

            constexpr int plane = W * W;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll 1
#endif
            for (int idx = team.team_rank(); idx < stencil_total; idx += team.team_size()) {
                const int zz  = idx / plane;
                const int rem = idx - zz * plane;
                const int yy  = rem / W;
                const int xx  = rem - yy * W;

                const int hidx   = (sx + xx) + (sy + yy) * pitch0 + (sz + zz) * pitch1;
                const RealType w = kwx[xx] * kwy[yy] * kwz[zz];
                local_r[hidx] += vr * w;
                if constexpr (NeedsImag)
                    local_i[hidx] += vi * w;
            }
        }

        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t           = typename decltype(args.grid)::non_const_value_type;
            constexpr bool grid_complex  = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool value_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
            constexpr bool needs_imag    = grid_complex && value_complex;

            const size_t league_r = static_cast<size_t>(team.league_rank());

            size_t tile_id, sub_id;
            if constexpr (fixed_oversubscription) {
                tile_id = league_r;
                sub_id  = 0;
            } else {
                tile_id = league_r / sub_teams_per_tile_;
                sub_id  = league_r - tile_id * sub_teams_per_tile_;
            }

            const size_t bin_start = args.bin_offsets(tile_id);
            const size_t bin_end   = args.bin_offsets(tile_id + 1);
            const size_t bin_size  = bin_end - bin_start;
            if (bin_size == 0)
                return;

            size_t active_subteams = 1;
            if constexpr (!fixed_oversubscription) {
                const size_t tgt = static_cast<size_t>(Kokkos::max(128, 4 * args.team_size));
                active_subteams  = Kokkos::min(sub_teams_per_tile_,
                                               Kokkos::max<size_t>(1, (bin_size + tgt - 1) / tgt));
                if (sub_id >= active_subteams)
                    return;
            }

            const size_t particles_per_sub = (bin_size + active_subteams - 1) / active_subteams;
            const size_t pstart            = bin_start + sub_id * particles_per_sub;
            if (pstart >= bin_end)
                return;
            const size_t pend = Kokkos::min(bin_end, pstart + particles_per_sub);

            const int hs0    = args.tile_size[0] + padded_extra;
            const int hs1    = args.tile_size[1] + padded_extra;
            const int pitch0 = hs0;
            const int pitch1 = hs0 * hs1;

            const size_t total = total_;
            const int batch_np = args.batch_np;

            int tile_base_x, tile_base_y, tile_base_z;
            decode_tile_base(tile_id, tile_base_x, tile_base_y, tile_base_z);

            auto scratch = team.team_scratch(0);

            scratch_real_view local_r_v(scratch, total);
            RealType* const local_r = local_r_v.data();

            RealType* local_i = nullptr;
            scratch_real_view local_i_v;
            if constexpr (needs_imag) {
                local_i_v = scratch_real_view(scratch, total);
                local_i   = local_i_v.data();
            }

            scratch_real_view kerevals_v(scratch, static_cast<size_t>(batch_np) * ker_stride);
            RealType* const kerevals = kerevals_v.data();

            scratch_real_view vals_r_v(scratch, batch_np);
            RealType* const vals_r = vals_r_v.data();

            RealType* vals_i = nullptr;
            scratch_real_view vals_i_v;
            if constexpr (needs_imag) {
                vals_i_v = scratch_real_view(scratch, batch_np);
                vals_i   = vals_i_v.data();
            }

            scratch_int_view shifts_x_v(scratch, batch_np);
            scratch_int_view shifts_y_v(scratch, batch_np);
            scratch_int_view shifts_z_v(scratch, batch_np);
            int* const shifts_x = shifts_x_v.data();
            int* const shifts_y = shifts_y_v.data();
            int* const shifts_z = shifts_z_v.data();

            for (size_t i = static_cast<size_t>(team.team_rank()); i < total;
                 i += static_cast<size_t>(team.team_size())) {
                local_r[i] = RealType(0);
                if constexpr (needs_imag)
                    local_i[i] = RealType(0);
            }
            team.team_barrier();

            const CoordinateTransform<RealType, 3> transform{args.origin, args.invdx, args.n_grid};

            for (size_t batch_begin = pstart; batch_begin < pend;
                 batch_begin += static_cast<size_t>(batch_np)) {
                const int batch_size = static_cast<int>(
                    Kokkos::min(pend - batch_begin, static_cast<size_t>(batch_np)));

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, batch_size), [&](const int bi) {
                    const size_t p = args.permute(batch_begin + static_cast<size_t>(bi));

                    const RealType gp0 = transform.template toGridCoordinate<0>(args.x(p)[0]);
                    const RealType gp1 = transform.template toGridCoordinate<1>(args.x(p)[1]);
                    const RealType gp2 = transform.template toGridCoordinate<2>(args.x(p)[2]);

                    const int idx0 = transform.template getStencilBase<W>(gp0 - RealType(0.5));
                    const int idx1 = transform.template getStencilBase<W>(gp1 - RealType(0.5));
                    const int idx2 = transform.template getStencilBase<W>(gp2 - RealType(0.5));

                    RealType* const kw = kerevals + static_cast<size_t>(bi) * ker_stride;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
                    for (int wi = 0; wi < W; ++wi) {
                        kw[wi]         = args.kernel((gp0 - (RealType(idx0 + wi) + RealType(0.5)))
                                                     * args.inv_hw);
                        kw[W + wi]     = args.kernel((gp1 - (RealType(idx1 + wi) + RealType(0.5)))
                                                     * args.inv_hw);
                        kw[2 * W + wi] = args.kernel((gp2 - (RealType(idx2 + wi) + RealType(0.5)))
                                                     * args.inv_hw);
                    }

                    const int sx = idx0 - args.local_offset[0] + half_left - tile_base_x;
                    const int sy = idx1 - args.local_offset[1] + half_left - tile_base_y;
                    const int sz = idx2 - args.local_offset[2] + half_left - tile_base_z;
                    shifts_x[bi] = sx;
                    shifts_y[bi] = sy;
                    shifts_z[bi] = sz;

                    if constexpr (value_complex) {
                        const auto v = args.values(p);
                        vals_r[bi]   = v.real();
                        if constexpr (needs_imag)
                            vals_i[bi] = v.imag();
                    } else {
                        vals_r[bi] = static_cast<RealType>(args.values(p));
                    }
                });
                team.team_barrier();

                for (int bi = 0; bi < batch_size; ++bi) {
                    const int sx             = shifts_x[bi];
                    const int sy             = shifts_y[bi];
                    const int sz             = shifts_z[bi];
                    const RealType* const kw = kerevals + static_cast<size_t>(bi) * ker_stride;
                    const RealType vr        = vals_r[bi];
                    const RealType vi        = needs_imag ? vals_i[bi] : RealType(0);

                    scatter_particle_fast<needs_imag>(team, sx, sy, sz, pitch0, pitch1, kw,
                                                      vr, vi, local_r, local_i);
                    // Per-particle barrier is required because consecutive
                    // particles' stencils can write to overlapping cells of
                    // the team-local histogram (`local_r` / `local_i`) using
                    // non-atomic `+=`. Without the barrier, two team threads
                    // working on different particles could race on the same
                    // cell.
                    team.team_barrier();
                }
            }

            for (size_t lid = static_cast<size_t>(team.team_rank()); lid < total;
                 lid += static_cast<size_t>(team.team_size())) {
                const int kz  = static_cast<int>(lid / static_cast<size_t>(pitch1));
                const int rem = static_cast<int>(lid) - kz * pitch1;
                const int jy  = rem / pitch0;
                const int ix  = rem - jy * pitch0;

                const RealType rr = local_r[lid];
                // Skip cells the team never touched (initialised to exactly 0
                // and never written). Real kernel evaluations of touched cells
                // are non-zero in finite precision, so the float `==` is safe.
                if constexpr (needs_imag) {
                    if (rr == RealType(0) && local_i[lid] == RealType(0))
                        continue;
                } else {
                    if (rr == RealType(0))
                        continue;
                }

                const int local_x = tile_base_x + ix - half_left;
                const int local_y = tile_base_y + jy - half_left;
                const int local_z = tile_base_z + kz - half_left;

                if (local_x < -args.nghost || local_x >= args.n_grid_local[0] + args.nghost
                    || local_y < -args.nghost || local_y >= args.n_grid_local[1] + args.nghost
                    || local_z < -args.nghost || local_z >= args.n_grid_local[2] + args.nghost)
                    continue;

                const int gx = local_x + args.nghost;
                const int gy = local_y + args.nghost;
                const int gz = local_z + args.nghost;

                if constexpr (grid_complex) {
                    RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gx, gy, gz));
                    Kokkos::atomic_add(&ptr[0], rr);
                    if constexpr (needs_imag)
                        Kokkos::atomic_add(&ptr[1], local_i[lid]);
                } else {
                    Kokkos::atomic_add(&args.grid(gx, gy, gz), static_cast<grid_value_t>(rr));
                }
            }
        }

        template <bool NeedsImag>
        static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int /*team_size*/,
                                           int z_batches) {
            const int batch_np = z_batches > 0 ? z_batches : 1;
            const int hs0      = tile_size[0] + padded_extra;
            const int hs1      = tile_size[1] + padded_extra;
            const int hs2      = tile_size[2] + padded_extra;
            const size_t total = static_cast<size_t>(hs0) * hs1 * hs2;

            size_t s = 0;
            s += scratch_real_view::shmem_size(total);
            if constexpr (NeedsImag)
                s += scratch_real_view::shmem_size(total);
            s += scratch_real_view::shmem_size(static_cast<size_t>(batch_np) * ker_stride);
            s += scratch_real_view::shmem_size(batch_np);
            if constexpr (NeedsImag)
                s += scratch_real_view::shmem_size(batch_np);
            s += scratch_int_view::shmem_size(batch_np);  // shifts_x
            s += scratch_int_view::shmem_size(batch_np);  // shifts_y
            s += scratch_int_view::shmem_size(batch_np);  // shifts_z
            return s;
        }

        void run(size_t n_particles) {
            using grid_value_t           = typename decltype(args.grid)::non_const_value_type;
            constexpr bool grid_complex  = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool value_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
            constexpr bool needs_imag    = grid_complex && value_complex;

            size_t n_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n_tiles *= static_cast<size_t>(args.num_tiles[d]);
            if (n_tiles == 0 || n_particles == 0)
                return;

            const int hs0 = args.tile_size[0] + padded_extra;
            const int hs1 = args.tile_size[1] + padded_extra;
            const int hs2 = args.tile_size[2] + padded_extra;
            total_        = static_cast<size_t>(hs0) * hs1 * hs2;

            size_t league_size;
            if constexpr (fixed_oversubscription) {
                sub_teams_per_tile_ = 1;
                league_size         = n_tiles;
            } else {
                sub_teams_per_tile_ =
                    std::max<size_t>(1, static_cast<size_t>(args.oversubscription_factor));
                league_size = n_tiles * sub_teams_per_tile_;
            }

            const size_t scratch =
                compute_scratch_size<needs_imag>(args.tile_size, args.team_size, args.batch_np);

            auto policy = team_policy(league_size, args.team_size, 1)
                              .set_scratch_size(0, Kokkos::PerTeam(scratch));
            Kokkos::parallel_for("GridParallelScatterOutputDrivenBatched3D", policy, *this);
        }
    };

    // -----------------------------------------------------------------------------
    // Generic Dim != 3 fallback
    // -----------------------------------------------------------------------------

    /*!
     * @struct GridParallelScatterImplND
     * @brief Generic-dimension fallback used when @c Types::Dim != 3.
     */
    template <int W, class Types, class Policy>
    struct GridParallelScatterImplND {
        static_assert(Policy::use_sorting,
                      "GridParallelScatter assumes sorted/bin-partitioned particles");
        static_assert(Types::Dim != 3, "GridParallelScatterImplND is the non-3D fallback");

        static constexpr bool requires_binning = true;
        static constexpr unsigned Dim          = Types::Dim;
        static constexpr int half_left         = (W + 1) / 2;
        static constexpr int padded_extra      = 2 * half_left;
        static constexpr int stencil_total     = int_pow(W, static_cast<int>(Dim));
        static constexpr bool fixed_oversubscription =
            GridParallelScatterTuning<Policy>::fixed_oversubscription;

        static constexpr int ker_stride = static_cast<int>(Dim) * W;

        using RealType        = typename Types::RealType;
        using ValueType       = typename Types::ValueType;
        using memory_space    = typename Types::memory_space;
        using execution_space = typename Types::execution_space;

        using team_policy   = Kokkos::TeamPolicy<execution_space>;
        using team_member   = typename team_policy::member_type;
        using scratch_space = typename execution_space::scratch_memory_space;

        using scratch_real_view =
            Kokkos::View<RealType*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using scratch_int_view =
            Kokkos::View<int*, scratch_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        struct Arguments : ScatterArgumentsBase<Arguments, Types> {
            Kokkos::View<size_t*, memory_space> permute;
            Kokkos::View<size_t*, memory_space> bin_offsets;
            Vector<int, Dim> num_tiles;
            Vector<int, Dim> tile_size;
            int team_size;
            int oversubscription_factor;
            int batch_np;

            template <class Field, class Positions, class Values, class Kernel>
            static Arguments create(Field& field, const Positions& pos, const Values& vals,
                                    const Kernel& k, const ScatterConfig<Dim>& config,
                                    const BinningResult<Dim, memory_space>& binning) {
                Arguments a;
                a.initBase(field, pos, vals, k);
                a.permute                 = binning.permute;
                a.bin_offsets             = binning.bin_offsets;
                a.num_tiles               = binning.num_tiles;
                a.tile_size               = config.get_tile_size();
                // Defensive: clamp team_size on host-only backends (see
                // TiledScatter Arguments::create for rationale).
#ifdef KOKKOS_ENABLE_SERIAL
                constexpr bool host_only =
                    std::is_same_v<execution_space, Kokkos::Serial>;
#else
                constexpr bool host_only = false;
#endif
                a.team_size               = host_only ? 1 : config.team_size;
                a.oversubscription_factor = config.oversubscription_factor;
                a.batch_np                = config.z_batches > 0 ? config.z_batches : 1;
                return a;
            }
        };

        Arguments args;
        size_t total_              = 1;
        size_t sub_teams_per_tile_ = 1;

        KOKKOS_INLINE_FUNCTION Vector<int, Dim> decode_tile_base(size_t tile_id) const {
            Vector<int, Dim> tile_base;
            for (size_t t = tile_id, d = Dim; d-- > 0;) {
                const int tc = static_cast<int>(t % static_cast<size_t>(args.num_tiles[d]));
                tile_base[d] = tc * args.tile_size[d];
                t /= static_cast<size_t>(args.num_tiles[d]);
            }
            return tile_base;
        }

        template <bool NeedsImag>
        KOKKOS_FORCEINLINE_FUNCTION static void scatter_particle_fast(
            const team_member& team, const int* __restrict__ shift,
            const size_t* __restrict__ stride, const RealType* __restrict__ kw, const RealType vr,
            const RealType vi, RealType* __restrict__ local_r, RealType* __restrict__ local_i) {
            for (int idx = team.team_rank(); idx < stencil_total; idx += team.team_size()) {
                int tmp     = idx;
                size_t hidx = 0;
                RealType w  = RealType(1);
                for (unsigned d = 0; d < Dim; ++d) {
                    const int wi = tmp % W;
                    tmp /= W;
                    hidx += static_cast<size_t>(shift[d] + wi) * stride[d];
                    w *= kw[d * W + wi];
                }
                local_r[hidx] += vr * w;
                if constexpr (NeedsImag)
                    local_i[hidx] += vi * w;
            }
        }

        KOKKOS_INLINE_FUNCTION void operator()(const team_member& team) const {
            using grid_value_t           = typename decltype(args.grid)::non_const_value_type;
            constexpr bool grid_complex  = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool value_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
            constexpr bool needs_imag    = grid_complex && value_complex;

            const size_t league_r = static_cast<size_t>(team.league_rank());

            size_t tile_id, sub_id;
            if constexpr (fixed_oversubscription) {
                tile_id = league_r;
                sub_id  = 0;
            } else {
                tile_id = league_r / sub_teams_per_tile_;
                sub_id  = league_r - tile_id * sub_teams_per_tile_;
            }

            const size_t bin_start = args.bin_offsets(tile_id);
            const size_t bin_end   = args.bin_offsets(tile_id + 1);
            const size_t bin_size  = bin_end - bin_start;
            if (bin_size == 0)
                return;

            size_t active_subteams = 1;
            if constexpr (!fixed_oversubscription) {
                const size_t tgt = static_cast<size_t>(Kokkos::max(128, 4 * args.team_size));
                active_subteams  = Kokkos::min(sub_teams_per_tile_,
                                               Kokkos::max<size_t>(1, (bin_size + tgt - 1) / tgt));
                if (sub_id >= active_subteams)
                    return;
            }

            const size_t particles_per_sub = (bin_size + active_subteams - 1) / active_subteams;
            const size_t pstart            = bin_start + sub_id * particles_per_sub;
            if (pstart >= bin_end)
                return;
            const size_t pend = Kokkos::min(bin_end, pstart + particles_per_sub);

            int hs[Dim];
            size_t stride[Dim];
            for (unsigned d = 0; d < Dim; ++d)
                hs[d] = args.tile_size[d] + padded_extra;
            stride[0] = 1;
            for (unsigned d = 1; d < Dim; ++d)
                stride[d] = stride[d - 1] * static_cast<size_t>(hs[d - 1]);

            const auto tile_base = decode_tile_base(tile_id);
            const size_t total   = total_;
            const int batch_np   = args.batch_np;

            auto scratch = team.team_scratch(0);

            scratch_real_view local_r_v(scratch, total);
            RealType* const local_r = local_r_v.data();

            RealType* local_i = nullptr;
            scratch_real_view local_i_v;
            if constexpr (needs_imag) {
                local_i_v = scratch_real_view(scratch, total);
                local_i   = local_i_v.data();
            }

            scratch_real_view kerevals_v(scratch, static_cast<size_t>(batch_np) * ker_stride);
            RealType* const kerevals = kerevals_v.data();

            scratch_real_view vals_r_v(scratch, batch_np);
            RealType* const vals_r = vals_r_v.data();

            RealType* vals_i = nullptr;
            scratch_real_view vals_i_v;
            if constexpr (needs_imag) {
                vals_i_v = scratch_real_view(scratch, batch_np);
                vals_i   = vals_i_v.data();
            }

            scratch_int_view shifts_v(scratch, static_cast<size_t>(batch_np) * Dim);
            int* const shifts = shifts_v.data();

            for (size_t i = static_cast<size_t>(team.team_rank()); i < total;
                 i += static_cast<size_t>(team.team_size())) {
                local_r[i] = RealType(0);
                if constexpr (needs_imag)
                    local_i[i] = RealType(0);
            }
            team.team_barrier();

            const CoordinateTransform<RealType, Dim> transform{args.origin, args.invdx,
                                                               args.n_grid};

            for (size_t batch_begin = pstart; batch_begin < pend;
                 batch_begin += static_cast<size_t>(batch_np)) {
                const int batch_size = static_cast<int>(
                    Kokkos::min(pend - batch_begin, static_cast<size_t>(batch_np)));

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, batch_size), [&](const int bi) {
                    const size_t p     = args.permute(batch_begin + static_cast<size_t>(bi));
                    RealType* const kw = kerevals + static_cast<size_t>(bi) * ker_stride;

                    for (unsigned d = 0; d < Dim; ++d) {
                        const RealType gp = transform.toGridCoordinate(args.x(p)[d], d);
                        const int idx0    = transform.getStencilBase(gp - RealType(0.5), W);
                        for (int wi = 0; wi < W; ++wi)
                            kw[d * W + wi] = args.kernel(
                                (gp - (RealType(idx0 + wi) + RealType(0.5))) * args.inv_hw);
                        shifts[static_cast<size_t>(bi) * Dim + d] =
                            idx0 - args.local_offset[d] + half_left - tile_base[d];
                    }

                    if constexpr (value_complex) {
                        const auto v = args.values(p);
                        vals_r[bi]   = v.real();
                        if constexpr (needs_imag)
                            vals_i[bi] = v.imag();
                    } else {
                        vals_r[bi] = static_cast<RealType>(args.values(p));
                    }
                });
                team.team_barrier();

                for (int bi = 0; bi < batch_size; ++bi) {
                    int shift[Dim];
                    for (unsigned d = 0; d < Dim; ++d)
                        shift[d] = shifts[static_cast<size_t>(bi) * Dim + d];
                    const RealType* const kw = kerevals + static_cast<size_t>(bi) * ker_stride;
                    const RealType vr        = vals_r[bi];
                    const RealType vi        = needs_imag ? vals_i[bi] : RealType(0);
                    scatter_particle_fast<needs_imag>(team, shift, stride, kw, vr, vi, local_r,
                                                      local_i);
                    team.team_barrier();
                }
            }

            for (size_t lid = static_cast<size_t>(team.team_rank()); lid < total;
                 lid += static_cast<size_t>(team.team_size())) {
                size_t tmp = lid;
                int coord[Dim];
                size_t sidx = 0;
                for (unsigned d = 0; d < Dim; ++d) {
                    coord[d] = static_cast<int>(tmp % static_cast<size_t>(hs[d]));
                    tmp /= static_cast<size_t>(hs[d]);
                    sidx += static_cast<size_t>(coord[d]) * stride[d];
                }

                const RealType rr = local_r[sidx];
                if constexpr (needs_imag) {
                    if (rr == RealType(0) && local_i[sidx] == RealType(0))
                        continue;
                } else {
                    if (rr == RealType(0))
                        continue;
                }

                Kokkos::Array<int, Dim> gc{};
                bool in_bounds = true;
                for (unsigned d = 0; d < Dim; ++d) {
                    const int local = tile_base[d] + coord[d] - half_left;
                    if (local < -args.nghost || local >= args.n_grid_local[d] + args.nghost) {
                        in_bounds = false;
                        break;
                    }
                    gc[d] = local + args.nghost;
                }
                if (!in_bounds)
                    continue;

                [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    if constexpr (grid_complex) {
                        RealType* ptr = reinterpret_cast<RealType*>(&args.grid(gc[Is]...));
                        Kokkos::atomic_add(&ptr[0], rr);
                        if constexpr (needs_imag)
                            Kokkos::atomic_add(&ptr[1], local_i[sidx]);
                    } else {
                        Kokkos::atomic_add(&args.grid(gc[Is]...), static_cast<grid_value_t>(rr));
                    }
                }(std::make_index_sequence<Dim>{});
            }
        }

        template <bool NeedsImag>
        static size_t compute_scratch_size(const Vector<int, Dim>& tile_size, int /*team_size*/,
                                           int z_batches) {
            const int batch_np = z_batches > 0 ? z_batches : 1;
            size_t total       = 1;
            for (unsigned d = 0; d < Dim; ++d)
                total *= static_cast<size_t>(tile_size[d] + padded_extra);

            size_t s = 0;
            s += scratch_real_view::shmem_size(total);
            if constexpr (NeedsImag)
                s += scratch_real_view::shmem_size(total);
            s += scratch_real_view::shmem_size(static_cast<size_t>(batch_np) * ker_stride);
            s += scratch_real_view::shmem_size(batch_np);
            if constexpr (NeedsImag)
                s += scratch_real_view::shmem_size(batch_np);
            s += scratch_int_view::shmem_size(static_cast<size_t>(batch_np) * Dim);
            return s;
        }

        void run(size_t n_particles) {
            using grid_value_t           = typename decltype(args.grid)::non_const_value_type;
            constexpr bool grid_complex  = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;
            constexpr bool value_complex = std::is_same_v<ValueType, Kokkos::complex<RealType>>;
            constexpr bool needs_imag    = grid_complex && value_complex;

            size_t n_tiles = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n_tiles *= static_cast<size_t>(args.num_tiles[d]);
            if (n_tiles == 0 || n_particles == 0)
                return;

            total_ = 1;
            for (unsigned d = 0; d < Dim; ++d)
                total_ *= static_cast<size_t>(args.tile_size[d] + padded_extra);

            size_t league_size;
            if constexpr (fixed_oversubscription) {
                sub_teams_per_tile_ = 1;
                league_size         = n_tiles;
            } else {
                sub_teams_per_tile_ =
                    std::max<size_t>(1, static_cast<size_t>(args.oversubscription_factor));
                league_size = n_tiles * sub_teams_per_tile_;
            }

            const size_t scratch =
                compute_scratch_size<needs_imag>(args.tile_size, args.team_size, args.batch_np);
            auto policy = team_policy(league_size, args.team_size, 1)
                              .set_scratch_size(0, Kokkos::PerTeam(scratch));
            Kokkos::parallel_for("GridParallelScatterOutputDrivenBatchedND", policy, *this);
        }
    };

    // -----------------------------------------------------------------------------
    // Public selector
    // -----------------------------------------------------------------------------

    template <int W, class Types, class Policy>
    using GridParallelScatter =
        std::conditional_t<(Types::Dim == 3), GridParallelScatterImpl3D<W, Types, Policy>,
                           GridParallelScatterImplND<W, Types, Policy>>;

}  // namespace ippl::Interpolation::detail

#endif  // IPPL_GRID_PARALLEL_SCATTER_H
