#ifndef ABC_H
#define ABC_H
#include <Kokkos_Core.hpp>

#include "Types/Vector.h"

#include "Field/Field.h"

template <unsigned a, unsigned b>
constexpr KOKKOS_INLINE_FUNCTION auto first() {
    return a;
}
template <unsigned a, unsigned b>
constexpr KOKKOS_INLINE_FUNCTION auto second() {
    return b;
}

/* @struct second_order_abc_face
 * @brief Implements a second-order absorbing boundary condition (ABC) for a face of a 3D
 * computational domain.
 *
 * This struct computes and applies the second-order Mur ABC on a specified face of the domain,
 * minimizing reflections for electromagnetic field solvers. The face is defined by its main axis
 * (the normal direction) and two side axes (the tangential directions).
 *
 * The struct precomputes weights based on mesh spacing and time step, and provides an operator()
 * to update the field at the boundary using current, previous, and next time-step values.
 *
 * @tparam _scalar      Scalar type (e.g., float or double).
 * @tparam _main_axis   The main axis (0=x, 1=y, 2=z) normal to the face.
 * @tparam _side_axes   The two side axes (tangential to the face).
 */
template <typename _scalar, unsigned _main_axis, unsigned... _side_axes>
struct second_order_abc_face {
    using scalar = _scalar;  // Define the scalar type for the weights and calculations
    scalar Cweights[5];  // Weights for the second-order ABC, precomputed based on mesh spacing and
                         // time step
    int sign;            // Sign of the main axis (1 for lower boundary, -1 for upper).
    constexpr static unsigned main_axis =
        _main_axis;  // Main axis normal to the face (0=x, 1=y, 2=z)

    /*!
     * @brief Constructor for the second-order ABC face.
     * @param hr Mesh spacing in each dimension.
     * @param dt Time step size.
     * @param _sign Sign of the main axis (1 or -1).
     */
    KOKKOS_FUNCTION second_order_abc_face(ippl::Vector<scalar, 3> hr, scalar dt, int _sign)
        : sign(_sign) {
        // Speed of light in the medium
        constexpr scalar c = 1;

        // Check that the main axis is one of the three axes and the side axes are different from
        // each other and from the main axis
        constexpr unsigned side_axes[2] = {_side_axes...};
        static_assert(
            (main_axis == 0 && first<_side_axes...>() == 1 && second<_side_axes...>() == 2)
            || (main_axis == 1 && first<_side_axes...>() == 0 && second<_side_axes...>() == 2)
            || (main_axis == 2 && first<_side_axes...>() == 0 && second<_side_axes...>() == 1));
        assert(_main_axis != side_axes[0]);
        assert(_main_axis != side_axes[1]);
        assert(side_axes[0] != side_axes[1]);

        // Calculate prefactor for the weights used to calculate A_np1 at the boundary
        scalar d = 1.0 / (2.0 * dt * hr[main_axis]) + 1.0 / (2.0 * c * dt * dt);

        // Calculate the weights for the boundary condition
        Cweights[0] = (1.0 / (2.0 * dt * hr[main_axis]) - 1.0 / (2.0 * c * dt * dt)) / d;
        Cweights[1] = (-1.0 / (2.0 * dt * hr[main_axis]) - 1.0 / (2.0 * c * dt * dt)) / d;
        assert(abs(Cweights[1] + 1) < 1e-6);  // Cweight[1] should always be -1
        Cweights[2] = (1.0 / (c * dt * dt) - c / (2.0 * hr[side_axes[0]] * hr[side_axes[0]])
                       - c / (2.0 * hr[side_axes[1]] * hr[side_axes[1]]))
                      / d;
        Cweights[3] = (c / (4.0 * hr[side_axes[0]] * hr[side_axes[0]])) / d;
        Cweights[4] = (c / (4.0 * hr[side_axes[1]] * hr[side_axes[1]])) / d;
    }

    /*!
     * @brief Applies the second-order ABC to the boundary of the field.
     * @param A_n Current time step field.
     * @param A_nm1 Previous time step field.
     * @param A_np1 Next time step field.
     * @param c Coordinates of the current point in the field.
     * @return Updated value at the boundary.
     */
    template <typename view_type, typename Coords>
    KOKKOS_INLINE_FUNCTION auto operator()(const view_type& A_n, const view_type& A_nm1,
                                           const view_type& A_np1, const Coords& c) const ->
        typename view_type::value_type {
        uint32_t i = c[0];
        uint32_t j = c[1];
        uint32_t k = c[2];
        using ippl::apply;
        constexpr unsigned side_axes[2] = {_side_axes...};

        // Create vectors with 1 in the direction of the specified axes and 0 in the others
        ippl::Vector<uint32_t, 3> side_axis1_onehot =
            ippl::Vector<uint32_t, 3>{side_axes[0] == 0, side_axes[0] == 1, side_axes[0] == 2};
        ippl::Vector<uint32_t, 3> side_axis2_onehot =
            ippl::Vector<uint32_t, 3>{side_axes[1] == 0, side_axes[1] == 1, side_axes[1] == 2};
        // Create a vector in the direction of the main axis, the sign is used to determine the
        // direction of the offset depending on which side of the boundary we are at
        ippl::Vector<uint32_t, 3> mainaxis_off = ippl::Vector<int32_t, 3>{
            (main_axis == 0) * sign, (main_axis == 1) * sign, (main_axis == 2) * sign};

        // Apply the boundary condition
        // The main axis is the one that is normal to the face, so we need to apply the boundary
        // condition in the direction of the main axis, this means we need to offset the coordinates
        // by the sign of the main axis
        return advanceBoundaryS(
            A_nm1(i, j, k), A_n(i, j, k), apply(A_nm1, c + mainaxis_off),
            apply(A_n, c + mainaxis_off), apply(A_np1, c + mainaxis_off),
            apply(A_n, c + side_axis1_onehot + mainaxis_off),
            apply(A_n, c - side_axis1_onehot + mainaxis_off),
            apply(A_n, c + side_axis2_onehot + mainaxis_off),
            apply(A_n, c - side_axis2_onehot + mainaxis_off), apply(A_n, c + side_axis1_onehot),
            apply(A_n, c - side_axis1_onehot), apply(A_n, c + side_axis2_onehot),
            apply(A_n, c - side_axis2_onehot));
    }

    /*!
     * @brief Advances the boundary condition using the precomputed weights.
     */
    template <typename value_type>
    KOKKOS_FUNCTION value_type advanceBoundaryS(const value_type& v1, const value_type& v2,
                                                const value_type& v3, const value_type& v4,
                                                const value_type& v5, const value_type& v6,
                                                const value_type& v7, const value_type& v8,
                                                const value_type& v9, const value_type& v10,
                                                const value_type& v11, const value_type& v12,
                                                const value_type& v13) const noexcept {
        value_type v0 = Cweights[0] * (v1 + v5) + (Cweights[1]) * v3 + (Cweights[2]) * (v2 + v4)
                        + (Cweights[3]) * (v6 + v7 + v10 + v11)
                        + (Cweights[4]) * (v8 + v9 + v12 + v13);
        return v0;
    }
};

/* @struct second_order_abc_edge
 * @brief Implements a second-order absorbing boundary condition (ABC) for an edge of a 3D
 * computational domain.
 *
 * This struct computes and applies the second-order Mur ABC on a specified edge of the domain,
 * minimizing reflections for electromagnetic field solvers. The edge is defined by its axis (the
 * edge direction) and two normal axes (the directions normal to the edge).
 *
 * The struct precomputes weights based on mesh spacing and time step, and provides an operator()
 * to update the field at the boundary using current, previous, and next time-step values.
 *
 * @tparam _scalar        Scalar type (e.g., float or double).
 * @tparam edge_axis      The axis (0=x, 1=y, 2=z) along the edge.
 * @tparam normal_axis1   The first normal axis to the edge.
 * @tparam normal_axis2   The second normal axis to the edge.
 * @tparam na1_zero       Boolean indicating direction for normal_axis1 (true for lower boundary,
 * false for upper).
 * @tparam na2_zero       Boolean indicating direction for normal_axis2 (true for lower boundary,
 * false for upper).
 */
template <typename _scalar, unsigned edge_axis, unsigned normal_axis1, unsigned normal_axis2,
          bool na1_zero, bool na2_zero>
struct second_order_abc_edge {
    using scalar = _scalar;  // Define the scalar type for the weights and calculations
    scalar Eweights[5];  // Weights for the second-order ABC, precomputed based on mesh spacing and
                         // time step

    /*!
     * @brief Constructor for the second-order ABC edge.
     * @param hr Mesh spacing in each dimension.
     * @param dt Time step size.
     */
    KOKKOS_FUNCTION second_order_abc_edge(ippl::Vector<scalar, 3> hr, scalar dt) {
        // Check that the edge axis is one of the three axes and the normal axes are different from
        // each other and from the edge axis
        static_assert(normal_axis1 != normal_axis2);
        static_assert(edge_axis != normal_axis2);
        static_assert(edge_axis != normal_axis1);
        // Check that the axes are in the correct order
        static_assert((edge_axis == 2 && normal_axis1 == 0 && normal_axis2 == 1)
                      || (edge_axis == 0 && normal_axis1 == 1 && normal_axis2 == 2)
                      || (edge_axis == 1 && normal_axis1 == 2 && normal_axis2 == 0));

        // Speed of light in the medium
        constexpr scalar c0_ = scalar(1);

        // Calculate prefactor for the weights used to calculate A_np1 at the boundary
        scalar d = (1.0 / hr[normal_axis1] + 1.0 / hr[normal_axis2]) / (4.0 * dt)
                   + 3.0 / (8.0 * c0_ * dt * dt);

        // Calculate the weights for the boundary condition
        Eweights[0] = (-(1.0 / hr[normal_axis2] - 1.0 / hr[normal_axis1]) / (4.0 * dt)
                       - 3.0 / (8.0 * c0_ * dt * dt))
                      / d;
        Eweights[1] = ((1.0 / hr[normal_axis2] - 1.0 / hr[normal_axis1]) / (4.0 * dt)
                       - 3.0 / (8.0 * c0_ * dt * dt))
                      / d;
        Eweights[2] = ((1.0 / hr[normal_axis2] + 1.0 / hr[normal_axis1]) / (4.0 * dt)
                       - 3.0 / (8.0 * c0_ * dt * dt))
                      / d;
        Eweights[3] =
            (3.0 / (4.0 * c0_ * dt * dt) - c0_ / (4.0 * hr[edge_axis] * hr[edge_axis])) / d;
        Eweights[4] = c0_ / (8.0 * hr[edge_axis] * hr[edge_axis]) / d;
    }

    /*!
     * @brief Applies the second-order ABC to the edge of the field.
     * @param A_n Current time step field.
     * @param A_nm1 Previous time step field.
     * @param A_np1 Next time step field.
     * @param c Coordinates of the current point in the field.
     * @return Updated value at the edge.
     */
    template <typename view_type, typename Coords>
    KOKKOS_INLINE_FUNCTION auto operator()(const view_type& A_n, const view_type& A_nm1,
                                           const view_type& A_np1, const Coords& c) const ->
        typename view_type::value_type {
        uint32_t i = c[0];
        uint32_t j = c[1];
        uint32_t k = c[2];
        using ippl::apply;

        // Get the opposite direction of the normal vectors for the two normal axes (direction to
        // the interior of the domain), 1 or -1 in the direction of the normal axis and 0 in the others
        ippl::Vector<int32_t, 3> normal_axis1_onehot =
            ippl::Vector<int32_t, 3>{normal_axis1 == 0, normal_axis1 == 1, normal_axis1 == 2}
            * int32_t(na1_zero ? 1 : -1);
        ippl::Vector<int32_t, 3> normal_axis2_onehot =
            ippl::Vector<int32_t, 3>{normal_axis2 == 0, normal_axis2 == 1, normal_axis2 == 2}
            * int32_t(na2_zero ? 1 : -1);

        // Calculate the coordinates of the neighboring points in the interior of the domain
        ippl::Vector<uint32_t, 3> acc0 = {i, j, k};
        ippl::Vector<uint32_t, 3> acc1 = acc0 + normal_axis1_onehot;
        ippl::Vector<uint32_t, 3> acc2 = acc0 + normal_axis2_onehot;
        ippl::Vector<uint32_t, 3> acc3 = acc0 + normal_axis1_onehot + normal_axis2_onehot;

        // Create a vector in the direction of the edge axis
        ippl::Vector<uint32_t, 3> axisp{edge_axis == 0, edge_axis == 1, edge_axis == 2};

        // Apply the boundary condition
        return advanceEdgeS(A_n(i, j, k), A_nm1(i, j, k), apply(A_np1, acc1), apply(A_n, acc1),
                            apply(A_nm1, acc1), apply(A_np1, acc2), apply(A_n, acc2),
                            apply(A_nm1, acc2), apply(A_np1, acc3), apply(A_n, acc3),
                            apply(A_nm1, acc3), apply(A_n, acc0 - axisp), apply(A_n, acc1 - axisp),
                            apply(A_n, acc2 - axisp), apply(A_n, acc3 - axisp),
                            apply(A_n, acc0 + axisp), apply(A_n, acc1 + axisp),
                            apply(A_n, acc2 + axisp), apply(A_n, acc3 + axisp));
    }

    /*!
     * @brief Advances the edge boundary condition using the precomputed weights.
     */
    template <typename value_type>
    KOKKOS_INLINE_FUNCTION value_type advanceEdgeS(value_type v1, value_type v2, value_type v3,
                                                   value_type v4, value_type v5, value_type v6,
                                                   value_type v7, value_type v8, value_type v9,
                                                   value_type v10, value_type v11, value_type v12,
                                                   value_type v13, value_type v14, value_type v15,
                                                   value_type v16, value_type v17, value_type v18,
                                                   value_type v19) const noexcept {
        value_type v0 = Eweights[0] * (v3 + v8) + Eweights[1] * (v5 + v6) + Eweights[2] * (v2 + v9)
                        + Eweights[3] * (v1 + v4 + v7 + v10)
                        + Eweights[4] * (v12 + v13 + v14 + v15 + v16 + v17 + v18 + v19) - v11;
        return v0;
    }
};

/* @struct second_order_abc_corner
 * @brief Implements a second-order absorbing boundary condition (ABC) for a corner of a 3D
 * computational domain.
 *
 * This struct computes and applies the second-order Mur ABC on a specified corner of the domain,
 * minimizing reflections for electromagnetic field solvers. The corner is defined by its three
 * axes, with each axis having a boolean indicating whether it is at the lower or upper boundary.
 *
 * The struct precomputes weights based on mesh spacing and time step, and provides an operator()
 * to update the field at the boundary using current, previous, and next time-step values.
 *
 * @tparam _scalar  Scalar type (e.g., float or double).
 * @tparam x0       Boolean indicating direction for x-axis (true for lower boundary, false for
 * upper).
 * @tparam y0       Boolean indicating direction for y-axis (true for lower boundary, false for
 * upper).
 * @tparam z0       Boolean indicating direction for z-axis (true for lower boundary, false for
 * upper).
 */
template <typename _scalar, bool x0, bool y0, bool z0>
struct second_order_abc_corner {
    using scalar = _scalar;  // Define the scalar type for the weights and calculations
    scalar Cweights[17];  // Weights for the second-order ABC, precomputed based on mesh spacing and
                          // time step

    /*!
     * @brief Constructor for the second-order ABC corner.
     * @param hr Mesh spacing in each dimension.
     * @param dt Time step size.
     */
    KOKKOS_FUNCTION second_order_abc_corner(ippl::Vector<scalar, 3> hr, scalar dt) {
        constexpr scalar c0_ = scalar(1);
        Cweights[0] =
            (-1.0 / hr[0] - 1.0 / hr[1] - 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[1] =
            (1.0 / hr[0] - 1.0 / hr[1] - 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[2] =
            (-1.0 / hr[0] + 1.0 / hr[1] - 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[3] =
            (-1.0 / hr[0] - 1.0 / hr[1] + 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[4] =
            (1.0 / hr[0] + 1.0 / hr[1] - 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[5] =
            (1.0 / hr[0] - 1.0 / hr[1] + 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[6] =
            (-1.0 / hr[0] + 1.0 / hr[1] + 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[7] =
            (1.0 / hr[0] + 1.0 / hr[1] + 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[8] =
            -(-1.0 / hr[0] - 1.0 / hr[1] - 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[9] =
            -(1.0 / hr[0] - 1.0 / hr[1] - 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[10] =
            -(-1.0 / hr[0] + 1.0 / hr[1] - 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[11] =
            -(-1.0 / hr[0] - 1.0 / hr[1] + 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[12] =
            -(1.0 / hr[0] + 1.0 / hr[1] - 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[13] =
            -(1.0 / hr[0] - 1.0 / hr[1] + 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[14] =
            -(-1.0 / hr[0] + 1.0 / hr[1] + 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[15] =
            -(1.0 / hr[0] + 1.0 / hr[1] + 1.0 / hr[2]) / (8.0 * dt) - 1.0 / (4.0 * c0_ * dt * dt);
        Cweights[16] = 1.0 / (2.0 * c0_ * dt * dt);
    }

    /*!
     * @brief Applies the second-order ABC to the corner of the field.
     * @param A_n Current time step field.
     * @param A_nm1 Previous time step field.
     * @param A_np1 Next time step field.
     * @param c Coordinates of the current point in the field.
     * @return Updated value at the corner.
     */
    template <typename view_type, typename Coords>
    KOKKOS_INLINE_FUNCTION auto operator()(const view_type& A_n, const view_type& A_nm1,
                                           const view_type& A_np1, const Coords& c) const ->
        typename view_type::value_type {
        // Get direction of the vector to the interior of the domain
        constexpr uint32_t xoff = (x0) ? 1 : uint32_t(-1);
        constexpr uint32_t yoff = (y0) ? 1 : uint32_t(-1);
        constexpr uint32_t zoff = (z0) ? 1 : uint32_t(-1);
        using ippl::apply;

        // Calculate the offset vectors for the neighboring points in the interior of the domain
        const ippl::Vector<uint32_t, 3> offsets[8] = {
            ippl::Vector<uint32_t, 3>{0, 0, 0},       ippl::Vector<uint32_t, 3>{xoff, 0, 0},
            ippl::Vector<uint32_t, 3>{0, yoff, 0},    ippl::Vector<uint32_t, 3>{0, 0, zoff},
            ippl::Vector<uint32_t, 3>{xoff, yoff, 0}, ippl::Vector<uint32_t, 3>{xoff, 0, zoff},
            ippl::Vector<uint32_t, 3>{0, yoff, zoff}, ippl::Vector<uint32_t, 3>{xoff, yoff, zoff},
        };

        // Apply the boundary condition
        return advanceCornerS(
            apply(A_n, c), apply(A_nm1, c), apply(A_np1, c + offsets[1]),
            apply(A_n, c + offsets[1]), apply(A_nm1, c + offsets[1]), apply(A_np1, c + offsets[2]),
            apply(A_n, c + offsets[2]), apply(A_nm1, c + offsets[2]), apply(A_np1, c + offsets[3]),
            apply(A_n, c + offsets[3]), apply(A_nm1, c + offsets[3]), apply(A_np1, c + offsets[4]),
            apply(A_n, c + offsets[4]), apply(A_nm1, c + offsets[4]), apply(A_np1, c + offsets[5]),
            apply(A_n, c + offsets[5]), apply(A_nm1, c + offsets[5]), apply(A_np1, c + offsets[6]),
            apply(A_n, c + offsets[6]), apply(A_nm1, c + offsets[6]), apply(A_np1, c + offsets[7]),
            apply(A_n, c + offsets[7]), apply(A_nm1, c + offsets[7]));
    }

    /*!
     * @brief Advances the corner boundary condition using the precomputed weights.
     */
    template <typename value_type>
    KOKKOS_INLINE_FUNCTION value_type
    advanceCornerS(value_type v1, value_type v2, value_type v3, value_type v4, value_type v5,
                   value_type v6, value_type v7, value_type v8, value_type v9, value_type v10,
                   value_type v11, value_type v12, value_type v13, value_type v14, value_type v15,
                   value_type v16, value_type v17, value_type v18, value_type v19, value_type v20,
                   value_type v21, value_type v22, value_type v23) const noexcept {
        return -(v1 * (Cweights[16]) + v2 * (Cweights[8]) + v3 * Cweights[1] + v4 * Cweights[16]
                 + v5 * Cweights[9] + v6 * Cweights[2] + v7 * Cweights[16] + v8 * Cweights[10]
                 + v9 * Cweights[3] + v10 * Cweights[16] + v11 * Cweights[11] + v12 * Cweights[4]
                 + v13 * Cweights[16] + v14 * Cweights[12] + v15 * Cweights[5] + v16 * Cweights[16]
                 + v17 * Cweights[13] + v18 * Cweights[6] + v19 * Cweights[16] + v20 * Cweights[14]
                 + v21 * Cweights[7] + v22 * Cweights[16] + v23 * Cweights[15])
               / Cweights[0];
    }
};

/* @struct second_order_mur_boundary_conditions
 * @brief Applies second-order Mur absorbing boundary conditions (ABC) to all boundaries of a 3D
 * computational domain.
 *
 * This struct coordinates the application of second-order Mur ABCs on faces, edges, and corners of
 * the domain, minimizing reflections for electromagnetic field solvers. It uses the appropriate
 * face, edge, or corner ABC implementation depending on the location of each boundary point.
 *
 * The apply() method detects the type of boundary (face, edge, or corner) for each point and
 * applies the corresponding second-order ABC using precomputed weights based on mesh spacing and
 * time step.
 *
 * @tparam field_type  The field type (must provide getView() and get_mesh().getMeshSpacing()).
 * @tparam dt_type     The type of the time step.
 */
struct second_order_mur_boundary_conditions {
    /*!
     * @brief Applies second-order Mur ABC to the boundaries of the field.
     * @param FA_n Field at the current time step.
     * @param FA_nm1 Field at the previous time step.
     * @param FA_np1 Field at the next time step.
     * @param dt Time step size.
     * @param true_nr True number of grid points in each dimension.
     * @param lDom Local domain index for parallel processing.
     */
    template <typename field_type, typename dt_type>
    void apply(field_type& FA_n, field_type& FA_nm1, field_type& FA_np1, dt_type dt,
               ippl::Vector<uint32_t, 3> true_nr, ippl::NDIndex<3> lDom) {
        using scalar = decltype(dt);

        // Get the mesh spacing
        const ippl::Vector<scalar, 3> hr = FA_n.get_mesh().getMeshSpacing();

        // Get views of the fields
        auto A_n   = FA_n.getView();
        auto A_np1 = FA_np1.getView();
        auto A_nm1 = FA_nm1.getView();

        // Get the local number of grid points in each dimension
        ippl::Vector<uint32_t, 3> local_nr{uint32_t(A_n.extent(0)), uint32_t(A_n.extent(1)),
                                           uint32_t(A_n.extent(2))};

        constexpr uint32_t min_abc_boundary     = 1;
        constexpr uint32_t max_abc_boundary_sub = min_abc_boundary + 1;

        // Apply second-order ABC to faces
        Kokkos::parallel_for(
            ippl::getRangePolicy(A_n, 1), KOKKOS_LAMBDA(uint32_t i, uint32_t j, uint32_t k) {
                // Get the global indices of the current point
                uint32_t ig = i + lDom.first()[0];
                uint32_t jg = j + lDom.first()[1];
                uint32_t kg = k + lDom.first()[2];

                // Check on how many boundaries of the local domain the current point is located
                uint32_t lval = uint32_t(i == 0) + (uint32_t(j == 0) << 1) + (uint32_t(k == 0) << 2)
                                + (uint32_t(i == local_nr[0] - 1) << 3)
                                + (uint32_t(j == local_nr[1] - 1) << 4)
                                + (uint32_t(k == local_nr[2] - 1) << 5);

                // Exits current iteration if the point is on an edge or corner of the local domain
                if (Kokkos::popcount(lval) > 1)
                    return;

                // Check on how many boundaries of the global domain the current point is located
                uint32_t val = uint32_t(ig == min_abc_boundary)
                               + (uint32_t(jg == min_abc_boundary) << 1)
                               + (uint32_t(kg == min_abc_boundary) << 2)
                               + (uint32_t(ig == true_nr[0] - max_abc_boundary_sub) << 3)
                               + (uint32_t(jg == true_nr[1] - max_abc_boundary_sub) << 4)
                               + (uint32_t(kg == true_nr[2] - max_abc_boundary_sub) << 5);

                // If the point is on a global face, apply the second-order ABC
                if (Kokkos::popcount(val) == 1) {
                    if (ig == min_abc_boundary) {
                        second_order_abc_face<scalar, 0, 1, 2> soa(hr, dt, 1);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    }
                    if (jg == min_abc_boundary) {
                        second_order_abc_face<scalar, 1, 0, 2> soa(hr, dt, 1);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    }
                    if (kg == min_abc_boundary) {
                        second_order_abc_face<scalar, 2, 0, 1> soa(hr, dt, 1);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    }
                    if (ig == true_nr[0] - max_abc_boundary_sub) {
                        second_order_abc_face<scalar, 0, 1, 2> soa(hr, dt, -1);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    }
                    if (jg == true_nr[1] - max_abc_boundary_sub) {
                        second_order_abc_face<scalar, 1, 0, 2> soa(hr, dt, -1);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    }
                    if (kg == true_nr[2] - max_abc_boundary_sub) {
                        second_order_abc_face<scalar, 2, 0, 1> soa(hr, dt, -1);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    }
                }
            });
        Kokkos::fence();

        // Apply second-order ABC to edges
        Kokkos::parallel_for(
            ippl::getRangePolicy(A_n, 1), KOKKOS_LAMBDA(uint32_t i, uint32_t j, uint32_t k) {
                // Get the global indices of the current point
                uint32_t ig = i + lDom.first()[0];
                uint32_t jg = j + lDom.first()[1];
                uint32_t kg = k + lDom.first()[2];

                // Check on how many boundaries of the local domain the current point is located
                uint32_t lval = uint32_t(i == 0) + (uint32_t(j == 0) << 1) + (uint32_t(k == 0) << 2)
                                + (uint32_t(i == local_nr[0] - 1) << 3)
                                + (uint32_t(j == local_nr[1] - 1) << 4)
                                + (uint32_t(k == local_nr[2] - 1) << 5);

                // Exits current iteration if the point is on a corner of the local domain
                if (Kokkos::popcount(lval) > 2)
                    return;

                // Check on how many boundaries of the global domain the current point is located
                uint32_t val = uint32_t(ig == min_abc_boundary)
                               + (uint32_t(jg == min_abc_boundary) << 1)
                               + (uint32_t(kg == min_abc_boundary) << 2)
                               + (uint32_t(ig == true_nr[0] - max_abc_boundary_sub) << 3)
                               + (uint32_t(jg == true_nr[1] - max_abc_boundary_sub) << 4)
                               + (uint32_t(kg == true_nr[2] - max_abc_boundary_sub) << 5);

                // If the point is on a global edge, apply the second-order ABC
                if (Kokkos::popcount(val) == 2) {
                    if (ig == min_abc_boundary && kg == min_abc_boundary) {
                        second_order_abc_edge<scalar, 1, 2, 0, true, true> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == min_abc_boundary && jg == min_abc_boundary) {
                        second_order_abc_edge<scalar, 2, 0, 1, true, true> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (jg == min_abc_boundary && kg == min_abc_boundary) {
                        second_order_abc_edge<scalar, 0, 1, 2, true, true> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == min_abc_boundary && kg == true_nr[2] - max_abc_boundary_sub) {
                        second_order_abc_edge<scalar, 1, 2, 0, false, true> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == min_abc_boundary && jg == true_nr[1] - max_abc_boundary_sub) {
                        second_order_abc_edge<scalar, 2, 0, 1, true, false> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (jg == min_abc_boundary && kg == true_nr[2] - max_abc_boundary_sub) {
                        second_order_abc_edge<scalar, 0, 1, 2, true, false> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == true_nr[0] - max_abc_boundary_sub && kg == min_abc_boundary) {
                        second_order_abc_edge<scalar, 1, 2, 0, true, false> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == true_nr[0] - max_abc_boundary_sub && jg == min_abc_boundary) {
                        second_order_abc_edge<scalar, 2, 0, 1, false, true> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (jg == true_nr[1] - max_abc_boundary_sub && kg == min_abc_boundary) {
                        second_order_abc_edge<scalar, 0, 1, 2, false, true> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == true_nr[0] - max_abc_boundary_sub
                               && kg == true_nr[2] - max_abc_boundary_sub) {
                        second_order_abc_edge<scalar, 1, 2, 0, false, false> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == true_nr[0] - max_abc_boundary_sub
                               && jg == true_nr[1] - max_abc_boundary_sub) {
                        second_order_abc_edge<scalar, 2, 0, 1, false, false> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (jg == true_nr[1] - max_abc_boundary_sub
                               && kg == true_nr[2] - max_abc_boundary_sub) {
                        second_order_abc_edge<scalar, 0, 1, 2, false, false> soa(hr, dt);
                        A_np1(i, j, k) = soa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else {
                        assert(false);
                    }
                }
            });
        Kokkos::fence();

        // Apply second-order ABC to corners
        Kokkos::parallel_for(
            ippl::getRangePolicy(A_n, 1), KOKKOS_LAMBDA(uint32_t i, uint32_t j, uint32_t k) {
                // Get the global indices of the current point
                uint32_t ig = i + lDom.first()[0];
                uint32_t jg = j + lDom.first()[1];
                uint32_t kg = k + lDom.first()[2];

                // Check on how many boundaries of the global domain the current point is located
                uint32_t val = uint32_t(ig == min_abc_boundary)
                               + (uint32_t(jg == min_abc_boundary) << 1)
                               + (uint32_t(kg == min_abc_boundary) << 2)
                               + (uint32_t(ig == true_nr[0] - max_abc_boundary_sub) << 3)
                               + (uint32_t(jg == true_nr[1] - max_abc_boundary_sub) << 4)
                               + (uint32_t(kg == true_nr[2] - max_abc_boundary_sub) << 5);

                // If the point is on a global corner, apply the second-order ABC
                if (Kokkos::popcount(val) == 3) {
                    if (ig == min_abc_boundary && jg == min_abc_boundary
                        && kg == min_abc_boundary) {
                        second_order_abc_corner<scalar, 1, 1, 1> coa(hr, dt);
                        A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == true_nr[0] - max_abc_boundary_sub && jg == min_abc_boundary
                               && kg == min_abc_boundary) {
                        second_order_abc_corner<scalar, 0, 1, 1> coa(hr, dt);
                        A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == min_abc_boundary && jg == true_nr[1] - max_abc_boundary_sub
                               && kg == min_abc_boundary) {
                        second_order_abc_corner<scalar, 1, 0, 1> coa(hr, dt);
                        A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == true_nr[0] - max_abc_boundary_sub
                               && jg == true_nr[1] - max_abc_boundary_sub
                               && kg == min_abc_boundary) {
                        second_order_abc_corner<scalar, 0, 0, 1> coa(hr, dt);
                        A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == min_abc_boundary && jg == min_abc_boundary
                               && kg == true_nr[2] - max_abc_boundary_sub) {
                        second_order_abc_corner<scalar, 1, 1, 0> coa(hr, dt);
                        A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == true_nr[0] - max_abc_boundary_sub && jg == min_abc_boundary
                               && kg == true_nr[2] - max_abc_boundary_sub) {
                        second_order_abc_corner<scalar, 0, 1, 0> coa(hr, dt);
                        A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == min_abc_boundary && jg == true_nr[1] - max_abc_boundary_sub
                               && kg == true_nr[2] - max_abc_boundary_sub) {
                        second_order_abc_corner<scalar, 1, 0, 0> coa(hr, dt);
                        A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else if (ig == true_nr[0] - max_abc_boundary_sub
                               && jg == true_nr[1] - max_abc_boundary_sub
                               && kg == true_nr[2] - max_abc_boundary_sub) {
                        second_order_abc_corner<scalar, 0, 0, 0> coa(hr, dt);
                        A_np1(i, j, k) = coa(A_n, A_nm1, A_np1, ippl::Vector<uint32_t, 3>{i, j, k});
                    } else {
                        assert(false);
                    }
                }
            });
        Kokkos::fence();
    }
};
#endif