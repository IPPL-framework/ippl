#ifndef IPPL_GATHER_CONFIG_H
#define IPPL_GATHER_CONFIG_H

#include <array>
#include <mutex>
#include <optional>

#include <Kokkos_Core.hpp>

#include "Types/Vector.h"

namespace ippl {
    namespace Interpolation {

        /**
         * @brief Gather method for grid-to-particle interpolation.
         *
         * - Atomic: read-only field gather, no sorting (no atomic ops actually
         *   needed since gather only reads — name kept for symmetry with the
         *   Scatter API).
         * - AtomicSort: same gather kernel, with a binning pre-pass that
         *   improves cache locality for clustered particle distributions.
         */
        enum class GatherMethod {
            Atomic,
            AtomicSort
        };

        struct GatherCacheEntry {
            GatherMethod       method = GatherMethod::Atomic;
            std::array<int, 3> tile   = {1, 1, 1};
        };

        class GatherCache {
        public:
            static GatherCache& instance() {
                static GatherCache c;
                std::call_once(c.once_, [&]() { c.load(); });
                return c;
            }

            std::optional<GatherCacheEntry> get() const {
                if (!loaded_) return std::nullopt;
                return entry_;
            }

            /// Seed a built-in default. Skipped if a CSV has already been
            /// loaded (CSV wins). Used by Ippl::initialize so users get a
            /// known-good config without paying the auto-tune cost.
            void seed_default(GatherMethod method, std::array<int, 3> tile) {
                std::call_once(once_, [&]() {
                    entry_  = GatherCacheEntry{method, tile};
                    loaded_ = true;
                });
            }

        private:
            void load();

            std::once_flag   once_;
            bool             loaded_ = false;
            GatherCacheEntry entry_;
        };

        /**
         * @brief Configuration for scatter/gather operations
         */
        template <unsigned Dim>
        struct GatherConfig {
            GatherMethod method = GatherMethod::Atomic;

            // Tile size per dimension
            std::array<int, Dim> tile_size;

            // Team size for team-based methods
            int team_size = 16;

            bool add_to_attribute = false;

            /**
             * @brief Default constructor - initializes tile sizes based on Dim
             */
            GatherConfig() {
                if constexpr (Dim == 1) {
                    tile_size.fill(512);
                } else if constexpr (Dim == 2) {
                    tile_size.fill(32);
                } else {
                    tile_size.fill(8);
                }
            }

            /**
             * @brief Constructor with uniform tile size for all dimensions
             */
            explicit GatherConfig(int uniform_tile_size)
                : GatherConfig() {
                tile_size.fill(uniform_tile_size);
            }

            /**
             * @brief Constructor with per-dimension tile sizes
             */
            explicit GatherConfig(std::array<int, Dim> tile_sizes)
                : GatherConfig() {
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
            GatherConfig& set_tile_size(int uniform_size) {
                tile_size.fill(uniform_size);
                return *this;
            }

            /**
             * @brief Set tile size per dimension
             */
            GatherConfig& set_tile_size(std::array<int, Dim> sizes) {
                tile_size = sizes;
                return *this;
            }

            /**
             * @brief Set tile size for a specific dimension
             */
            GatherConfig& set_tile_size(unsigned dim, int size) {
                tile_size[dim] = size;
                return *this;
            }

            bool do_binning() const {
                return method == GatherMethod::AtomicSort;
            }

            /**
             * @brief Get default configuration for an execution space
             */
            template <typename ExecSpace>
            static GatherConfig get_default();
        };

        // Helper to define defaults for execution space + dimension combinations
        namespace detail {
            template <unsigned Dim, typename ExecSpace, typename = void>
            struct GatherConfigDefault {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method = GatherMethod::Atomic;
                    return config;
                }
            };

#ifdef KOKKOS_ENABLE_SERIAL
            template <unsigned Dim>
            struct GatherConfigDefault<Dim, Kokkos::Serial> {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method = GatherMethod::Atomic;
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_CUDA
            template <unsigned Dim>
            struct GatherConfigDefault<Dim, Kokkos::Cuda> {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method    = GatherMethod::AtomicSort;
                    config.team_size = 32;

                    if constexpr (Dim == 1) {
                        config.tile_size = {512};
                    } else if constexpr (Dim == 2) {
                        config.tile_size = {16, 16};
                    } else if constexpr (Dim == 3) {
                        config.tile_size = {4, 4, 4};
                    }
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_OPENMP
            template <unsigned Dim>
            struct GatherConfigDefault<Dim, Kokkos::OpenMP> {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method    = GatherMethod::Atomic;
                    config.team_size = 1;

                    if constexpr (Dim == 1) {
                        config.tile_size = {256};
                    } else if constexpr (Dim == 2) {
                        config.tile_size = {16, 16};
                    } else if constexpr (Dim == 3) {
                        config.tile_size = {9, 9, 9};
                    }
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_HIP
            // Wavefronts are 64-wide (vs CUDA's 32) so a larger tile per
            // team keeps occupancy similar; team_size = wavefront width.
            template <unsigned Dim>
            struct GatherConfigDefault<Dim, Kokkos::HIP> {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method    = GatherMethod::AtomicSort;
                    config.team_size = 64;

                    if constexpr (Dim == 1) {
                        config.tile_size = {512};
                    } else if constexpr (Dim == 2) {
                        config.tile_size = {16, 16};
                    } else if constexpr (Dim == 3) {
                        config.tile_size = {6, 6, 6};
                    }
                    return config;
                }
            };
#endif

#ifdef KOKKOS_ENABLE_THREADS
            template <unsigned Dim>
            struct GatherConfigDefault<Dim, Kokkos::Threads> {
                static GatherConfig<Dim> get() {
                    GatherConfig<Dim> config;
                    config.method = GatherMethod::AtomicSort;
                    return config;
                }
            };
#endif
        }  // namespace detail

        // Implementation of get_default using the helper. Overrides the
        // backend baseline with the AutoTune-recorded entry when present.
        template <unsigned Dim>
        template <typename ExecSpace>
        GatherConfig<Dim> GatherConfig<Dim>::get_default() {
            GatherConfig<Dim> cfg = detail::GatherConfigDefault<Dim, ExecSpace>::get();
            if (auto cached = GatherCache::instance().get()) {
                cfg.method = cached->method;
                if constexpr (Dim <= 3) {
                    for (unsigned d = 0; d < Dim; ++d) {
                        cfg.tile_size[d] = cached->tile[d];
                    }
                }
            }
            return cfg;
        }

    }  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_GATHER_CONFIG_H
