#ifndef IPPL_INTERPOLATION_COORDINATE_TRANSFORM_H
#define IPPL_INTERPOLATION_COORDINATE_TRANSFORM_H

#include <Kokkos_Core.hpp>

#include "Types/Vector.h"

#include "Index/Index.h"

namespace ippl::Interpolation {

    // Forward declare size types from InterpolationUtil.h
    using local_index_type = decltype(Index{}.first());
    using size_type        = ippl::detail::size_type;

    /**
     * @brief Unified coordinate transformation for scatter/gather operations
     *
     * This class handles the transformation from physical coordinates to grid
     * coordinates using mesh information (origin and spacing).
     *
     * @tparam T Floating-point type for coordinates
     * @tparam Dim Spatial dimension
     */
    template <typename T, unsigned Dim>
    struct CoordinateTransform {
        using Vector_t    = ippl::Vector<T, Dim>;
        using VectorInt_t = ippl::Vector<int, Dim>;

        const Vector_t origin_;           // Physical origin from mesh
        const Vector_t invdx_;            // Inverse of mesh spacing (1/dx)
        const VectorInt_t ngrid_global_;  // Global grid dimensions

        /**
         * @brief Construct from mesh parameters
         *
         * @param origin Physical origin of the domain
         * @param invdx Inverse mesh spacing (1/dx)
         * @param ngrid_global Global grid dimensions
         */
        KOKKOS_INLINE_FUNCTION CoordinateTransform(const Vector_t& origin, const Vector_t& invdx,
                                                   const VectorInt_t& ngrid_global)
            : origin_(origin)
            , invdx_(invdx)
            , ngrid_global_(ngrid_global) {}

        /**
         * @brief Transform physical position to grid coordinates [0, ngrid)
         *
         * This function scales from physical domain to grid domain: (x - origin) / dx
         *
         * @param physical_pos Physical position in the specified dimension
         * @param dim Dimension index
         * @return Grid coordinate in [0, ngrid)
         */
        KOKKOS_INLINE_FUNCTION T toGridCoordinate(T physical_pos, unsigned dim) const {
            return (physical_pos - origin_[dim]) * invdx_[dim];
        }

        template <int D>
        KOKKOS_FORCEINLINE_FUNCTION T toGridCoordinate(T physical_pos) const {
            return (physical_pos - origin_[D]) * invdx_[D];
        }

        /**
         * @brief Round a grid coordinate to the cell index that anchors the stencil.
         *
         * Uses width-dependent rounding:
         * - Odd width:  round to nearest (symmetric stencil around the particle)
         * - Even width: floor             (asymmetric stencil)
         *
         * The stencil leftmost cell is then `center - (width - 1) / 2`, computed
         * by `getStencilBase`.
         *
         * @param grid_pos Grid coordinate (output of toGridCoordinate)
         * @param width Kernel width
         * @return Center cell index
         */
        KOKKOS_INLINE_FUNCTION int getStencilCenter(T grid_pos, int width) const {
            const bool odd = (width & 1);
            return odd ? static_cast<int>(Kokkos::round(grid_pos))
                       : static_cast<int>(Kokkos::floor(grid_pos));
        }

        template <int Width>
        KOKKOS_FORCEINLINE_FUNCTION int getStencilCenter(T grid_pos) const {
            if constexpr (Width & 1)
                return static_cast<int>(Kokkos::round(grid_pos));
            else
                return static_cast<int>(Kokkos::floor(grid_pos));
        }

        /**
         * @brief Get base grid index for kernel stencil
         *
         * Uses width-dependent rounding to determine the base index:
         * - Odd width: round to nearest (symmetric stencil around particle)
         * - Even width: floor (asymmetric stencil)
         *
         * The stencil extends from [base_idx, base_idx + width).
         *
         * For odd widths (e.g., w=3):
         *   - Grid point at 2.7 rounds to 3
         *   - Stencil covers indices [3-(3-1)/2, 3+(3-1)/2] = [2, 3, 4]
         *
         * For even widths (e.g., w=4):
         *   - Grid point at 2.7 floors to 2
         *   - Stencil covers indices [2-(4-1)/2, 2+(4-1)/2+1] = [0, 1, 2, 3]
         *
         * @param grid_pos Grid coordinate (output of toGridCoordinate)
         * @param width Kernel width
         * @return Base index for the kernel stencil (leftmost index)
         */
        KOKKOS_INLINE_FUNCTION int getStencilBase(T grid_pos, int width) const {
            return getStencilCenter(grid_pos, width) - (width - 1) / 2;
        }

        template <int Width>
        KOKKOS_FORCEINLINE_FUNCTION int getStencilBase(T grid_pos) const {
            return getStencilCenter<Width>(grid_pos) - (Width - 1) / 2;
        }
    };

}  // namespace ippl::Interpolation

#endif  // IPPL_INTERPOLATION_COORDINATE_TRANSFORM_H
