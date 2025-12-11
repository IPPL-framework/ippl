#ifndef IPPL_SCATTER_CONFIG_H
#define IPPL_SCATTER_CONFIG_H

namespace ippl {
namespace Interpolation {

    /**
     * @brief Scattering/gathering method for particle-grid interpolation
     *
     * Different methods offer various performance characteristics:
     * - Atomic: Simple atomic operations, works everywhere
     * - Tiled: Cache-friendly tiling with team policies and shared memory histograms
     */
    enum class ScatterMethod {
        Atomic,        // Simple atomic operations
        Tiled,         // Tiled for cache locality with shared memory optimization
        OutputFocused
    };

    /**
     * @brief Configuration for scatter/gather operations
     */
    struct ScatterConfig {
        ScatterMethod method = ScatterMethod::Atomic;
        bool sort = false;  // Sort particles by spatial location before scattering

        // Tile size for tiled methods (per dimension)
        int tile_size_1d = 512;
        int tile_size_2d = 32;
        int tile_size_3d = 16;

        // Z-dimension splitting for 3D tiled scatter (reduces shared memory)
        int z_tiles = 2;

        // Team size for team-based methods
        int team_size = 16;

        /**
         * @brief Get default configuration for an execution space
         */
        template <typename ExecSpace>
        static ScatterConfig get_default();
    };

    // Default configurations for different execution spaces
#ifdef KOKKOS_ENABLE_SERIAL
    template <>
    inline ScatterConfig ScatterConfig::get_default<Kokkos::Serial>() {
        ScatterConfig config;
        config.method = ScatterMethod::Atomic;
        config.sort = false;
        return config;
    }
#endif

#ifdef KOKKOS_ENABLE_CUDA
    template <>
    inline ScatterConfig ScatterConfig::get_default<Kokkos::Cuda>() {
        ScatterConfig config;
        config.method = ScatterMethod::OutputFocused;
        config.sort = true;
        config.tile_size_3d = 3;
        config.z_tiles = 4;
        config.team_size = 32;
        return config;
    }
#endif


#ifdef KOKKOS_ENABLE_OPENMP
    template <>
    inline ScatterConfig ScatterConfig::get_default<Kokkos::OpenMP>() {
        ScatterConfig config;
        config.method = ScatterMethod::Atomic;
        config.sort = false;
        config.tile_size_3d = 9;
        config.team_size = 1;
        return config;
    }
#endif


#ifdef KOKKOS_ENABLE_HIP
    template <>
    inline ScatterConfig ScatterConfig::get_default<Kokkos::HIP>() {
        ScatterConfig config;
        config.method = ScatterMethod::Tiled;
        config.sort = true;
        config.tile_size_3d = 6;
        config.z_tiles = 2;
        config.team_size = 64;
        return config;
    }
#endif

#ifdef KOKKOS_ENABLE_THREADS
    template <>
    inline ScatterConfig ScatterConfig::get_default<Kokkos::Threads>() {
        ScatterConfig config;
        config.method = ScatterMethod::Atomic;
        config.sort = true;
        return config;
    }
#endif

}  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_SCATTER_CONFIG_H
