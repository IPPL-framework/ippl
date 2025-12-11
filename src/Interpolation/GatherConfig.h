#ifndef IPPL_GATHER_CONFIG_H
#define IPPL_GATHER_CONFIG_H

namespace ippl {
namespace Interpolation {

    /**
     * @brief Scattering/gathering method for particle-grid interpolation
     *
     * Different methods offer various performance characteristics:
     * - Atomic: Simple atomic operations, works everywhere
     * - Tiled: Cache-friendly tiling with team policies and shared memory histograms
     */
    enum class GatherMethod {
        Atomic,
        AtomicSort,
        Tiled,
        Native
    };


    /**
     * @brief Configuration for scatter/gather operations
     */
    struct GatherConfig {
        GatherMethod method = GatherMethod::Tiled;
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
        static GatherConfig get_default();
    };

    // Default configurations for different execution spaces
#ifdef KOKKOS_ENABLE_SERIAL
    template <>
    inline GatherConfig GatherConfig::get_default<Kokkos::Serial>() {
        GatherConfig config;
        config.method = GatherMethod::Atomic;
        config.sort = false;
        return config;
    }
#endif

#ifdef KOKKOS_ENABLE_CUDA
    template <>
    inline GatherConfig GatherConfig::get_default<Kokkos::Cuda>() {
        GatherConfig config;
        config.method = GatherMethod::Tiled;
        config.sort = true;
        config.tile_size_3d = 4;
        config.z_tiles = 6;
        config.team_size = 32;
        return config;
    }
#endif


#ifdef KOKKOS_ENABLE_OPENMP
    template <>
    inline GatherConfig GatherConfig::get_default<Kokkos::OpenMP>() {
        GatherConfig config;
        config.method = GatherMethod::Atomic;
        config.sort = false;
        config.tile_size_3d = 9;
        config.team_size = 1;
        return config;
    }
#endif


#ifdef KOKKOS_ENABLE_HIP
    template <>
    inline GatherConfig GatherConfig::get_default<Kokkos::HIP>() {
        GatherConfig config;
        config.method = GatherMethod::Tiled;
        config.sort = true;
        config.tile_size_3d = 6;
        config.z_tiles = 2;
        config.team_size = 64;
        return config;
    }
#endif

#ifdef KOKKOS_ENABLE_THREADS
    template <>
    inline GatherConfig GatherConfig::get_default<Kokkos::Threads>() {
        GatherConfig config;
        config.method = GatherMethod::Atomic;
        config.sort = true;
        return config;
    }
#endif

}  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_SCATTER_CONFIG_H
