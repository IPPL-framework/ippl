#ifndef IPPL_SCATTER_CONFIG_H
#define IPPL_SCATTER_CONFIG_H

#include <array>

#include <Kokkos_Core.hpp>

#include "Types/Vector.h"

namespace ippl {
    namespace Interpolation {

        /**
         * @brief Scatter algorithm for particle → grid interpolation.
         *
         * - Atomic:        per-particle stencil scatter with `Kokkos::atomic_add`.
         * - Tiled:         bin particles into tiles, scatter each tile's
         *                  histogram into team-local scratch, then flush.
         * - OutputFocused: same as Tiled but with grid-parallel histogram
         *                  filling (one team per output tile, batched
         *                  particle reads).
         */
        enum class ScatterMethod {
            Atomic,
            Tiled,
            OutputFocused
        };

        /**
         * @brief Configuration for scatter operations
         */
        template <unsigned Dim>
        struct ScatterConfig {
            ScatterMethod method = ScatterMethod::Atomic;
            bool sort            = false;
            bool enable_tuning   = false;

            /// When true, TileSizeCache auto-selection will not override `method`.
            /// Tile sizes, team_size, osub, and z_batches are still loaded from
            /// the cache for the explicitly-chosen method.
            bool lock_method = false;

            // Tile size per dimension
            std::array<int, Dim> tile_size;

            // Team size for team-based methods
            int team_size = 1;

            // Factor for Gridparallel
            int oversubscription_factor = 4;

            // Number of z-stencil batches for GridParallelScatter.
            // When > 1, the kernel is launched multiple times, each time
            // processing only ceil(W/z_batches) z-stencil points. This reduces
            // shared memory pressure at the cost of multiple kernel launches.
            // Default 1 means no batching (process all z-stencil points at once).
            int z_batches = 1;

            /**
             * @brief Default constructor - initializes tile sizes based on Dim.
             *
             * 3-D defaults are intentionally conservative: the shared-memory
             * footprint of the scatter histogram grows as
             *   (tile + W + 1)^3  *  sizeof(RealType)  *  (2 if complex)
             * For the largest supported kernel width (W = 14) and complex<double>:
             *   (2 + 14 + 1)^3 * 8 * 2 = 17^3 * 16 ≈ 78 KB  — fits in 96 KB L1.
             *   (3 + 14 + 1)^3 * 8 * 2 = 18^3 * 16 ≈ 93 KB  — marginal.
             * Tile = 2 is therefore the safe 3-D default.  Scatter::dispatch()
             * further reduces the tile at runtime if the resolved config would
             * still exceed the device's available shared memory.
             */
            ScatterConfig() {
                if constexpr (Dim == 1) {
                    tile_size.fill(512);
                } else if constexpr (Dim == 2) {
                    tile_size.fill(16);
                } else {
                    // Dim == 3: conservative default (see comment above)
                    tile_size.fill(2);
                }
            }

            /**
             * @brief Constructor with uniform tile size for all dimensions
             */
            explicit ScatterConfig(int uniform_tile_size)
                : ScatterConfig() {
                tile_size.fill(uniform_tile_size);
            }

            /**
             * @brief Constructor with per-dimension tile sizes
             */
            explicit ScatterConfig(std::array<int, Dim> tile_sizes)
                : ScatterConfig() {
                tile_size = tile_sizes;
            }

            /**
             * @brief Get tile size as Vector for use in kernels
             */
            Vector<int, Dim> get_tile_size() const {
                Vector<int, Dim> result;
                for (unsigned d = 0; d < Dim; ++d) {
                    result[d] = tile_size[d];
                }
                return result;
            }

            /**
             * @brief Set uniform tile size for all dimensions
             */
            ScatterConfig& set_tile_size(int uniform_size) {
                tile_size.fill(uniform_size);
                return *this;
            }

            void set_tile_size(const Vector<int, Dim>& tile) {
                for (unsigned d = 0; d < Dim; ++d) {
                    tile_size[d] = tile[d];
                }
            }

            /**
             * @brief Set tile size per dimension
             */
            ScatterConfig& set_tile_size(std::array<int, Dim> sizes) {
                tile_size = sizes;
                return *this;
            }

            /**
             * @brief Set tile size for a specific dimension
             */
            ScatterConfig& set_tile_size(unsigned dim, int size) {
                tile_size[dim] = size;
                return *this;
            }

            bool do_binning() const { return method != ScatterMethod::Atomic || sort; }

            /**
             * @brief Get default configuration for an execution space
             */
            template <typename ExecSpace>
            static ScatterConfig get_default();
        };

        // Helper to define defaults for execution space + dimension combinations
        namespace detail {
            template <unsigned Dim, typename ExecSpace, typename = void>
            struct ScatterConfigDefault {
                static ScatterConfig<Dim> get() {
                    ScatterConfig<Dim> config;
                    config.method = ScatterMethod::Atomic;
                    config.sort   = false;
                    return config;
                }
            };

#ifdef KOKKOS_ENABLE_SERIAL
            template <unsigned Dim>
            struct ScatterConfigDefault<Dim, Kokkos::Serial> {
                static ScatterConfig<Dim> get() {
                    ScatterConfig<Dim> config;
                    config.method = ScatterMethod::Atomic;
                    config.sort   = false;
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_CUDA
            template <unsigned Dim>
            struct ScatterConfigDefault<Dim, Kokkos::Cuda> {
                static ScatterConfig<Dim> get() {
                    ScatterConfig<Dim> config;
                    config.method    = ScatterMethod::Atomic;
                    config.sort      = false;
                    config.team_size = 32;

                    if constexpr (Dim == 1) {
                        config.tile_size = {512};
                    } else if constexpr (Dim == 2) {
                        config.tile_size = {16, 16};
                    } else {
                        // Dim == 3: conservative default (see ScatterConfig() comment)
                        config.tile_size = {2, 2, 2};
                    }
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_OPENMP
            template <unsigned Dim>
            struct ScatterConfigDefault<Dim, Kokkos::OpenMP> {
                static ScatterConfig<Dim> get() {
                    ScatterConfig<Dim> config;
                    config.method    = ScatterMethod::Atomic;
                    config.sort      = false;
                    config.team_size = 1;

                    if constexpr (Dim == 1) {
                        config.tile_size = {256};
                    } else if constexpr (Dim == 2) {
                        config.tile_size = {16, 16};
                    } else {
                        config.tile_size = {9, 9, 9};
                    }
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_HIP
            template <unsigned Dim>
            struct ScatterConfigDefault<Dim, Kokkos::HIP> {
                static ScatterConfig<Dim> get() {
                    // CDNA wavefronts are 64-wide and the on-chip atomic
                    // throughput is more contention-sensitive than NVIDIA's,
                    // so the binned Tiled path tends to win as the default
                    // (with team_size = wavefront for full occupancy). The
                    // AutoTune pre-pass overrides these whenever it has a
                    // measured value for the running architecture.
                    ScatterConfig<Dim> config;
                    config.method    = ScatterMethod::Tiled;
                    config.sort      = true;
                    config.team_size = 64;

                    if constexpr (Dim == 1) {
                        config.tile_size = {512};
                    } else if constexpr (Dim == 2) {
                        config.tile_size = {16, 16};
                    } else {
                        // Dim == 3: conservative (same shmem rationale as CUDA)
                        config.tile_size = {2, 2, 2};
                    }
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_THREADS
            template <unsigned Dim>
            struct ScatterConfigDefault<Dim, Kokkos::Threads> {
                static ScatterConfig<Dim> get() {
                    ScatterConfig<Dim> config;
                    config.method    = ScatterMethod::Atomic;
                    config.sort      = true;
                    config.team_size = 1;

                    if constexpr (Dim == 1) {
                        config.tile_size = {256};
                    } else if constexpr (Dim == 2) {
                        config.tile_size = {16, 16};
                    } else {
                        config.tile_size = {9, 9, 9};
                    }
                    return config;
                }
            };
#endif
        }  // namespace detail

        // Implementation of get_default using the helper
        template <unsigned Dim>
        template <typename ExecSpace>
        ScatterConfig<Dim> ScatterConfig<Dim>::get_default() {
            return detail::ScatterConfigDefault<Dim, ExecSpace>::get();
        }

    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_SCATTER_CONFIG_H