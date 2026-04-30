#ifndef IPPL_SCATTER_H
#define IPPL_SCATTER_H

#include "Utility/Tuning.h"

#include "Interpolation/Binning.h"
#include "Interpolation/Scatter/AtomicScatter.h"
#include "Interpolation/Scatter/GridParallelScatter.h"
#include "Interpolation/Scatter/ScatterArgumentsBase.h"
#include "Interpolation/Scatter/ScatterConfig.h"
#include "Interpolation/Scatter/TileSizeCache.h"
#include "Interpolation/Scatter/TiledScatter.h"
#include "Interpolation/WidthDispatcher.h"
#include "Particle/ParticleAttrib.h"

namespace ippl {

    namespace Interpolation::detail {

        struct UnsortedPolicy {
            static constexpr bool use_sorting      = false;
            static constexpr bool requires_binning = false;
        };

        struct SortedPolicy {
            static constexpr bool use_sorting      = true;
            static constexpr bool requires_binning = true;
        };

        template <typename Kernel, typename FieldType, typename PositionsType, typename ValuesType>
        struct DeduceScatterTypes {
            using FieldTr = ippl::detail::FieldTraits<std::decay_t<FieldType>>;
            using PosTr   = ippl::detail::AttribTraits<std::decay_t<PositionsType>>;
            using ValTr   = ippl::detail::AttribTraits<std::decay_t<ValuesType>>;

            // RealType from the kernel's value_type — see Gather.h for the
            // rationale (avoids float-position downcasting the mesh spacing).
            using RealType = typename std::decay_t<Kernel>::value_type;

            using type = ScatterTypes<FieldTr::dim, RealType, std::decay_t<Kernel>,
                                      typename FieldTr::view_type, typename PosTr::view_type,
                                      typename ValTr::view_type>;
        };

        template <typename Kernel, typename FieldType, typename PositionsType, typename ValuesType>
        using DeducedScatterTypes =
            typename DeduceScatterTypes<Kernel, FieldType, PositionsType, ValuesType>::type;

        template <template <int, class, class> class Impl, unsigned Dim, typename RealType,
                   bool IsComplex>
        TileSizeTuner<Dim, Vector<int, Dim>>& get_scatter_tuner() {
            // The IsComplex template parameter is only used as part of the
            // function template's mangled name, ensuring real and complex
            // tuners stay separate without sharing a static-local instance.
            static TileSizeTuner<Dim, Vector<int, Dim>> tuner;
            return tuner;
        }

    }  // namespace Interpolation::detail

    namespace detail {
        template <typename T>
        struct is_kokkos_complex : std::false_type {};
        template <typename T>
        struct is_kokkos_complex<Kokkos::complex<T>> : std::true_type {};
    }  // namespace detail

    template <typename Kernel, unsigned Dim>
    class Scatter {
    public:
        Scatter(const Kernel& kernel, const Interpolation::ScatterConfig<Dim>& config = {})
            : kernel_m(kernel)
            , config_m(config) {}

        /**
         * @brief Scatter `values` from `positions` into `field`.
         *
         * Side effects:
         *  - `field` is overwritten (zeroed and refilled with the scatter sum).
         *    Callers wanting to accumulate must keep their own running buffer.
         *  - `field.accumulateHalo()` is invoked at the end (one MPI exchange).
         *  - When `enable_tuning=false` and `lock_method=false`, the resolved
         *    method may be re-selected from `TileSizeCache::get_best(...)`,
         *    which mutates `config_m.method` for subsequent calls.
         */
        template <typename ValueT, typename FieldT, class Mesh, class Centering, class... ViewArgs,
                  typename ParticleT, class... PosProps, class... ValProps>
        void operator()(Field<FieldT, Dim, Mesh, Centering, ViewArgs...>& field,
                        const ParticleAttrib<Vector<ParticleT, Dim>, PosProps...>& positions,
                        const ParticleAttrib<ValueT, ValProps...>& values) {
            using Types =
                Interpolation::detail::DeducedScatterTypes<Kernel, decltype(field),
                                                           decltype(positions), decltype(values)>;

            constexpr bool is_complex_field = ippl::detail::is_kokkos_complex<FieldT>::value;

            // ── Estimate particle density (rho = local particles / local grid pts) ──
            const size_t n_particles = positions.getParticleCount();
            const double rho_est     = estimate_rho(field, n_particles);

            // Auto-select the best method from the benchmark cache when not
            // tuning and not locked. Tile/team/osub/z_batches are applied later
            // in resolve_config for all methods, whether auto-selected or
            // explicitly locked.
            if (!config_m.enable_tuning && !config_m.lock_method) {
                auto& cache = Interpolation::TileSizeCache::instance();
                if (cache.loaded()) {
                    if (auto best = cache.get_best(kernel_m.width(), is_complex_field, rho_est)) {
                        config_m.method = best->method;
                    }
                }
            }

            // Backend-feasibility clamp. CSV presets are tagged by GPU arch but
            // a runtime mismatch (e.g. an sm_90 CSV being read by a Serial
            // build) leaves the cache holding Tiled/OutputFocused configs with
            // team_size > 1 - those abort inside their TeamPolicy construction
            // on a host backend. Force Atomic on backends where teams are not
            // a meaningful parallelism (Serial today; extend the trait below
            // if more host-only backends ever land).
#ifdef KOKKOS_ENABLE_SERIAL
            using exec_space_ = typename Types::execution_space;
#endif
            constexpr bool host_only_backend =
#ifdef KOKKOS_ENABLE_SERIAL
                std::is_same_v<exec_space_, Kokkos::Serial>
#else
                false
#endif
                ;
            if constexpr (host_only_backend) {
                if (config_m.method != Interpolation::ScatterMethod::Atomic) {
                    config_m.method = Interpolation::ScatterMethod::Atomic;
                }
            }

            const auto method = config_m.method;

            const bool run_atomic = (method == Interpolation::ScatterMethod::Atomic);

            if (run_atomic) {
                if (config_m.sort) {
                    dispatch<Interpolation::detail::AtomicScatter, Types,
                             Interpolation::detail::SortedPolicy>(field, positions, values,
                                                                  rho_est);
                } else {
                    dispatch<Interpolation::detail::AtomicScatter, Types,
                             Interpolation::detail::UnsortedPolicy>(field, positions, values,
                                                                    rho_est);
                }
                return;
            }

            if (method == Interpolation::ScatterMethod::Tiled) {
                dispatch<Interpolation::detail::TiledScatter, Types,
                         Interpolation::detail::SortedPolicy>(field, positions, values, rho_est);
                return;
            }

            if (method == Interpolation::ScatterMethod::OutputFocused) {
                dispatch<Interpolation::detail::GridParallelScatter, Types,
                         Interpolation::detail::SortedPolicy>(field, positions, values, rho_est);
            }
        }

    private:
        // ------------------------------------------------------------------
        // estimate_rho: LOCAL particles / LOCAL grid points.
        //
        // Uses getLocalNDIndex() for the rank-local owned domain, matching
        // getParticleCount() which returns the rank-local particle count.
        // The previous version used getDomain() (global) with local particle
        // counts, under-estimating rho by a factor of #ranks.
        // ------------------------------------------------------------------
        template <typename FieldT, class Mesh, class Centering, class... ViewArgs>
        static double estimate_rho(const Field<FieldT, Dim, Mesh, Centering, ViewArgs...>& field,
                                   size_t n_particles) {
            const auto& local_dom = field.getLayout().getLocalNDIndex();
            size_t n_grid         = 1;
            for (unsigned d = 0; d < Dim; ++d)
                n_grid *= static_cast<size_t>(local_dom[d].length());
            return (n_grid > 0) ? static_cast<double>(n_particles) / static_cast<double>(n_grid)
                                : 1.0;
        }

        // ------------------------------------------------------------------
        // resolve_config: load cached config for the ACTIVE method.
        //
        // Works identically for lock_method=true and lock_method=false:
        // by this point config_m.method is already set (either kept as-is
        // when locked, or switched by get_best() in operator()).
        //
        // Priority:
        //   1. enable_tuning → return config_m unchanged (runtime tuner)
        //   2. cache hit     → apply tile, team_size, osub, z_batches
        //   3. no hit        → return config_m unchanged
        // ------------------------------------------------------------------
        template <template <int, class, class> class Impl, int W, class Types, class Policy,
                  bool IsComplex>
        Interpolation::ScatterConfig<Dim> resolve_config(double rho_est) const {
            if (config_m.enable_tuning)
                return config_m;

            auto& cache = Interpolation::TileSizeCache::instance();
            auto cached = cache.get(config_m.method, W, IsComplex, rho_est);

            if (!cached.has_value())
                return config_m;

            const auto& e = cached.value();

            auto resolved = config_m;
            Vector<int, Dim> tile;
            for (unsigned d = 0; d < Dim; ++d)
                tile[d] = e.tile[d < 3 ? d : 2];
            resolved.set_tile_size(tile);
            if (e.team_size > 0)
                resolved.team_size = e.team_size;
            if (e.oversubscription_factor > 0)
                resolved.oversubscription_factor = e.oversubscription_factor;
            if (e.z_batches > 0)
                resolved.z_batches = e.z_batches;
            return resolved;
        }

        // ------------------------------------------------------------------
        // clamp_tile_to_shmem
        // ------------------------------------------------------------------
        template <template <int, class, class> class Impl, int W, class Types, class Policy,
                  bool IsComplex>
        static void clamp_tile_to_shmem(Interpolation::ScatterConfig<Dim>& cfg) {
            if constexpr (!Impl<W, Types, Policy>::requires_binning)
                return;

            using execution_space = typename Types::execution_space;
            using team_policy     = Kokkos::TeamPolicy<execution_space>;

            Vector<int, Dim> tile = cfg.get_tile_size();

            // Per-team scratch capacity is invariant across the loop iterations
            // for a fixed team_size. Hoist the query.
            const size_t avail = team_policy(1, cfg.team_size).scratch_size_max(0);

            const int max_iter = static_cast<int>(Dim) * 64 + 8;
            for (int itr = 0; itr < max_iter; ++itr) {
                const size_t req = Impl<W, Types, Policy>::template compute_scratch_size<IsComplex>(
                    tile, cfg.team_size, cfg.z_batches);
                if (req <= avail)
                    break;

                int best_dim    = -1;
                size_t best_req = req;

                for (unsigned d = 0; d < Dim; ++d) {
                    if (tile[d] <= 1)
                        continue;
                    Vector<int, Dim> trial = tile;
                    --trial[d];
                    size_t trial_req =
                        Impl<W, Types, Policy>::template compute_scratch_size<IsComplex>(
                            trial, cfg.team_size, cfg.z_batches);
                    if (trial_req < best_req) {
                        best_req = trial_req;
                        best_dim = static_cast<int>(d);
                    }
                }

                if (best_dim < 0) {
                    if (cfg.team_size > 1) {
                        cfg.team_size = std::max(1, cfg.team_size / 2);
                    } else {
                        break;
                    }
                } else {
                    --tile[best_dim];
                }
            }

            cfg.set_tile_size(tile);
        }

        // ------------------------------------------------------------------
        // dispatch
        // ------------------------------------------------------------------
        template <template <int, class, class> class Impl, class Types, class Policy, class Field,
                  class Positions, class Values>
        void dispatch(Field& field, const Positions& positions, const Values& values,
                      double rho_est) {
            using memory_space = typename Types::memory_space;
            using RealType     = typename Types::RealType;
            using grid_value_t = typename Field::value_type;

            constexpr bool is_complex = std::is_same_v<grid_value_t, Kokkos::complex<RealType>>;

            const int width          = kernel_m.width();
            const size_t n_particles = positions.getParticleCount();

            Interpolation::WidthDispatcher<1, std::decay_t<decltype(kernel_m)>::max_width>::dispatch(width, [&]<int W>() {
                // ── Step 1: Resolve config from cache (density-aware) ──────────
                auto tuned_config = resolve_config<Impl, W, Types, Policy, is_complex>(rho_est);

                if constexpr (Impl<W, Types, Policy>::requires_binning) {
                    if (config_m.enable_tuning) {
                        Vector<int, Dim> tuned_tile =
                            get_tuned_tile_size<Impl, W, Types, Policy, is_complex>(
                                field, tuned_config.get_tile_size());
                        tuned_config.set_tile_size(tuned_tile);
                    }
                }

                // ── Step 1b: Safety clamp (asymmetric, dimension-aware) ────────
                clamp_tile_to_shmem<Impl, W, Types, Policy, is_complex>(tuned_config);

                const Vector<int, Dim> tile_size = tuned_config.get_tile_size();

                // ── Step 2: Binning ────────────────────────────────────────────
                Interpolation::detail::BinningResult<Dim, memory_space> binning;
                if constexpr (Impl<W, Types, Policy>::requires_binning) {
                    binning = performBinning<Types>(positions, field, tile_size);
                } else if (config_m.do_binning()) {
                    binning = performBinning<Types>(positions, field, tile_size);
                }

                // ── Step 3: Run functor ────────────────────────────────────────
                auto args = Impl<W, Types, Policy>::Arguments::create(
                    field, positions, values, kernel_m, tuned_config, binning);
                Impl<W, Types, Policy> functor{std::move(args)};

                field = 0.0;

                functor.run(n_particles);
                Kokkos::fence();

                // ── Step 4: End tuning context ─────────────────────────────────
                if constexpr (Impl<W, Types, Policy>::requires_binning) {
                    if (config_m.enable_tuning) {
                        auto& tuner = Interpolation::detail::get_scatter_tuner<Impl, Dim, RealType,
                                                                               is_complex>();
                        tuner.end();
                    }
                }

                field.accumulateHalo();
            });
        }

        template <template <int, class, class> class Impl, int W, class Types, class Policy,
                  bool IsComplex, class Field>
        Vector<int, Dim> get_tuned_tile_size(const Field& /*field*/,
                                             const Vector<int, Dim>& default_tile) {
            using RealType        = typename Types::RealType;
            using execution_space = typename Types::execution_space;
            using team_policy     = Kokkos::TeamPolicy<execution_space>;

            auto& tuner =
                Interpolation::detail::get_scatter_tuner<Impl, Dim, RealType, IsComplex>();

            if (!tuner.is_initialized()) {
                const size_t max_scratch = team_policy(1, config_m.team_size).scratch_size_max(0);
                std::vector<int> candidates = {1, 2, 3, 4, 8, 16, 32};

                auto scratch_calc = [&](const Vector<int, Dim>& tile) {
                    return Impl<W, Types, Policy>::template compute_scratch_size<IsComplex>(
                        tile, config_m.team_size, config_m.z_batches);
                };
                tuner.initialize("Scatter_" + std::string(typeid(Impl<W, Types, Policy>).name()),
                                 candidates, max_scratch, scratch_calc, default_tile);
            }
            return tuner.begin();
        }

        template <typename Types, typename Positions, typename Field>
        auto performBinning(const Positions& positions, const Field& field,
                            const Vector<int, Dim>& tile_size) {
            using memory_space                     = typename Types::memory_space;
            auto [permute, bin_offsets, num_tiles] = Interpolation::detail::bin_particles(
                positions, field.getLayout(), field.get_mesh(), tile_size, kernel_m.width());

            return Interpolation::detail::BinningResult<Dim, memory_space>{permute, bin_offsets,
                                                                           num_tiles};
        }

        Kernel kernel_m;
        Interpolation::ScatterConfig<Dim> config_m;
    };

}  // namespace ippl

#endif  // IPPL_SCATTER_H