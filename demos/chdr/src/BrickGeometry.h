/** @file BrickGeometry.h
 * @brief Analytic axis-aligned brick predicate for staircase material assignment.
 * @ingroup chdr_geometry
 */
#ifndef IPPL_CHDR_BRICK_GEOMETRY_H
#define IPPL_CHDR_BRICK_GEOMETRY_H

#include <Kokkos_Core.hpp>

#include <type_traits>

#include "ChdrMeshConfig.h"

namespace chdr {

    /** @brief Axis-aligned dielectric brick point classification for host and device.
     * @ingroup chdr_geometry
     *
     * A brick is specified by its lower corner and positive edge lengths, in metres.
     * With lower corner @f$\vec{\ell}@f$, edge lengths @f$\mathbf L@f$ and
     * upper corner @f$\mathbf u=\vec{\ell}+\mathbf L@f$, contains() evaluates
     * the six face inequalities
     * @f[\ell_d-\tau\le p_d\le u_d+\tau,\qquad d=0,1,2.@f]
     * The length tolerance @f$\tau=64\epsilon_{\rm mach}s@f$ is defined by
     * mesh_config_detail::validateBrick(). Boundary points belong to the dielectric;
     * the tolerance absorbs coordinate roundoff, not a finite interface thickness.
     *
     * Construction and volume evaluation run on the host; contains() is callable
     * on host and device. Only doubles are stored, so a value copy can be captured
     * in an IPPL/Kokkos kernel without allocations, virtual dispatch or references
     * to host memory. There is no rotation parameter: choose PrismGeometry for a
     * triangular prism with an arbitrary extrusion axis.
     */
    class BrickGeometry {
    public:
        /** @brief Construct on the host from a configuration's brick fields.
         * @param config Configuration whose brickLower and brickSize are in metres.
         * @pre The caller selects this class only for GeometryType::Brick; this
         * overload uses brick fields and does not dispatch on config.geometryType.
         * @throws std::runtime_error If brick geometry validation fails.
         * @note Domain containment is the responsibility of readMeshConfig().
         */
        explicit BrickGeometry(const MeshConfig& config)
            : BrickGeometry(config.brickLower, config.brickSize) {}

        /** @brief Validate dimensions and precompute bounds and volume on the host.
         * @param lower Finite lower-corner coordinates (x,y,z), in metres.
         * @param size Finite positive lengths (x,y,z), in metres.
         * @throws std::runtime_error For invalid dimensions, unrepresentable upper
         * faces or nonfinite/nonpositive volume; see mesh_config_detail::validateBrick().
         * @note This constructor is independent of any computational-domain bounds.
         */
        BrickGeometry(const std::array<double, 3>& lower, const std::array<double, 3>& size) {
            tolerance_m        = mesh_config_detail::validateBrick(lower, size);
            long double volume = 1.0L;
            for (int d = 0; d < 3; ++d) {
                lower_m[d] = lower[d];
                upper_m[d] = lower[d] + size[d];
                volume *= static_cast<long double>(size[d]);
            }
            volume_m = static_cast<double>(volume);
        }

        /** @brief Evaluate the six brick face inequalities on host or device.
         * @param x Laboratory x coordinate, in metres.
         * @param y Laboratory y coordinate, in metres.
         * @param z Laboratory z coordinate, in metres.
         * @return True inside the brick, including boundary points within the
         * roundoff tolerance; false if any face inequality fails.
         * @note The caller supplies physical global cell-centre coordinates to
         * form a staircase discretization. No mesh spacing is used by this predicate.
         */
        KOKKOS_INLINE_FUNCTION bool contains(double x, double y, double z) const {
            const double point[3] = {x, y, z};
            for (int d = 0; d < 3; ++d) {
                if (!(point[d] >= lower_m[d] - tolerance_m && point[d] <= upper_m[d] + tolerance_m))
                    return false;
            }
            return true;
        }

        /** @brief Return the precomputed analytic volume on the host.
         * @return @f$V=L_xL_yL_z@f$, in cubic metres, computed from the input lengths.
         * This reference can be compared with the occupied staircase-cell volume.
         */
        double volume() const { return volume_m; }

    private:
        double lower_m[3]{};       ///< Lower-corner coordinates, in metres.
        double upper_m[3]{};       ///< Upper-corner coordinates, in metres.
        double tolerance_m = 0.0;  ///< Boundary roundoff tolerance, in metres.
        double volume_m    = 0.0;  ///< Analytic volume, in cubic metres.
    };

    static_assert(std::is_trivially_copyable_v<BrickGeometry>,
                  "BrickGeometry must remain safe to copy into a device kernel");

}  // namespace chdr

#endif
