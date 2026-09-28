/** @file PrismGeometry.h
 * @brief Analytic triangular-prism predicate used to assign staircase material cells.
 * @ingroup chdr_geometry
 */
#ifndef IPPL_CHDR_PRISM_GEOMETRY_H
#define IPPL_CHDR_PRISM_GEOMETRY_H

#include <Kokkos_Core.hpp>

#include <type_traits>

#include "ChdrMeshConfig.h"

namespace chdr {

    /** @brief Convex triangular prism point classification for host and device.
     * @ingroup chdr_geometry
     *
     * A triangle with vertices @f$\mathbf v_i@f$, @f$i=0,1,2@f$, is extruded a
     * distance @f$H>0@f$ along its perpendicular unit axis @f$\hat{\mathbf a}@f$.
     * Both vertex windings and arbitrary extrusion directions are supported.
     * Host construction validates the inputs, normalizes the already nearly unit
     * axis and computes unit inward normals @f$\hat{\mathbf n}_i@f$ for the sides.
     * For @f$\mathbf p=(x,y,z)@f$, contains() tests the five face inequalities
     * @f[
     * -\tau\le(\mathbf p-\mathbf v_0)\cdot\hat{\mathbf a}\le H+\tau,
     * \qquad (\mathbf p-\mathbf v_i)\cdot\hat{\mathbf n}_i\ge-\tau
     * \quad(i=0,1,2).
     * @f]
     * The length tolerance @f$\tau=64\epsilon_{\rm mach}s@f$ is defined by
     * mesh_config_detail::validatePrism(). It absorbs coordinate roundoff and
     * assigns boundary points to the radiator; it does not smooth the interface.
     *
     * Only doubles are stored. Copy the object by value into an IPPL/Kokkos kernel
     * to classify physical cell centres without allocations, virtual dispatch or
     * host memory references. Constructors and volume() are host-only; contains()
     * is callable on host and device. No field update or boundary condition is
     * applied by this geometry class.
     */
    class PrismGeometry {
    public:
        /** @brief Construct on the host from a configuration's prism fields.
         * @param config Configuration whose vertices, axis and height are in SI units.
         * @pre The caller selects this class only for GeometryType::Prism; this
         * overload uses prism fields and does not dispatch on config.geometryType.
         * @throws std::runtime_error If prism geometry validation fails.
         * @note Domain containment is the responsibility of readMeshConfig().
         */
        explicit PrismGeometry(const MeshConfig& config)
            : PrismGeometry(config.vertices, config.axis, config.height) {}

        /** @brief Validate and precompute the five face tests on the host.
         * @param vertices Three noncollinear bottom-face coordinates, in metres.
         * @param axis Perpendicular dimensionless direction, unit length within 1e-12.
         * @param height Positive extrusion length, in metres.
         * @throws std::runtime_error For invalid, nonfinite or degenerate geometry;
         * see mesh_config_detail::validatePrism() for the complete constraints.
         *
         * For edge @f$\mathbf e_i=\mathbf v_{(i+1)\bmod 3}-\mathbf v_i@f$,
         * the side normal is
         * @f$\hat{\mathbf n}_i=q(\hat{\mathbf a}\times\mathbf e_i)/
         * |\hat{\mathbf a}\times\mathbf e_i|@f$.
         * The winding sign @f$q@f$ is positive when
         * @f$[(\mathbf v_1-\mathbf v_0)\times(\mathbf v_2-\mathbf v_0)]
         * \cdot\hat{\mathbf a}\ge0@f$ and negative otherwise.
         */
        PrismGeometry(const std::array<std::array<double, 3>, 3>& vertices,
                      const std::array<double, 3>& axis, double height) {
            namespace detail    = mesh_config_detail;
            tolerance_m         = detail::validatePrism(vertices, axis, height);
            height_m            = height;
            const auto axisNorm = std::sqrt(detail::dot(axis, axis));
            std::array<double, 3> unitAxis{};
            std::array<double, 3> edge0{}, edge1{};
            for (int d = 0; d < 3; ++d) {
                unitAxis[d] = axis[d] / axisNorm;
                axis_m[d]   = unitAxis[d];
                edge0[d]    = vertices[1][d] - vertices[0][d];
                edge1[d]    = vertices[2][d] - vertices[0][d];
                for (int i = 0; i < 3; ++i)
                    vertices_m[i][d] = vertices[i][d];
            }
            const auto faceNormal = detail::cross(edge0, edge1);
            volume_m              = 0.5 * std::sqrt(detail::dot(faceNormal, faceNormal)) * height;
            const double orientation = detail::dot(faceNormal, unitAxis) >= 0.0 ? 1.0 : -1.0;
            for (int i = 0; i < 3; ++i) {
                std::array<double, 3> edge{};
                for (int d = 0; d < 3; ++d)
                    edge[d] = vertices[(i + 1) % 3][d] - vertices[i][d];
                const auto inward   = detail::cross(unitAxis, edge);
                const double length = std::sqrt(detail::dot(inward, inward));
                for (int d = 0; d < 3; ++d)
                    inwardNormals_m[i][d] = orientation * inward[d] / length;
            }
        }

        /** @brief Evaluate the five prism face inequalities on host or device.
         * @param x Laboratory x coordinate, in metres.
         * @param y Laboratory y coordinate, in metres.
         * @param z Laboratory z coordinate, in metres.
         * @return True inside the prism, including faces within the roundoff
         * tolerance; false if any face inequality fails.
         * @note This is a point predicate. The caller supplies global cell-centre
         * coordinates for a staircase discretization; no mesh spacing is needed here.
         */
        KOKKOS_INLINE_FUNCTION bool contains(double x, double y, double z) const {
            const double point[3] = {x, y, z};
            double longitudinal   = 0.0;
            for (int d = 0; d < 3; ++d) {
                longitudinal += (point[d] - vertices_m[0][d]) * axis_m[d];
            }
            if (!(longitudinal >= -tolerance_m && longitudinal <= height_m + tolerance_m)) {
                return false;
            }
            for (int i = 0; i < 3; ++i) {
                double distance = 0.0;
                for (int d = 0; d < 3; ++d) {
                    distance += (point[d] - vertices_m[i][d]) * inwardNormals_m[i][d];
                }
                if (!(distance >= -tolerance_m))
                    return false;
            }
            return true;
        }

        /** @brief Return the precomputed analytic volume on the host.
         * @return @f$V=\frac12|(\mathbf v_1-\mathbf v_0)\times
         * (\mathbf v_2-\mathbf v_0)|H@f$, in cubic metres. This is an analytic
         * reference for comparison with the volume of occupied staircase cells.
         */
        double volume() const { return volume_m; }

    private:
        double vertices_m[3][3]{};       ///< Bottom-face vertices, in metres.
        double axis_m[3]{};              ///< Normalized, dimensionless extrusion direction.
        double inwardNormals_m[3][3]{};  ///< Unit inward side normals, one per triangle edge.
        double height_m    = 0.0;        ///< Extrusion length, in metres.
        double tolerance_m = 0.0;        ///< Boundary roundoff tolerance, in metres.
        double volume_m    = 0.0;        ///< Analytic volume, in cubic metres.
    };

    static_assert(std::is_trivially_copyable_v<PrismGeometry>,
                  "PrismGeometry must remain safe to copy into a device kernel");

}  // namespace chdr

#endif
