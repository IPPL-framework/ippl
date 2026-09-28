/** @file datatypes.h
 * @brief Common mesh, particle, collocated-field and solver types for the FEL mini-app.
 * @ingroup fel_fields
 *
 * All field aliases use the same UniformCartesian mesh and its default cell
 * centering. The active solver advances the collocated four-potential and then
 * reconstructs electric and magnetic fields. These aliases do not define a Yee
 * staggered mesh, a dielectric material model or a nonuniform coordinate spacing.
 */
#ifndef IPPL_FEL_DATATYPES_H
#define IPPL_FEL_DATATYPES_H

#include "MaxwellSolvers/StandardFDTDSolver.h"
#include "MaxwellSolvers/NonStandardFDTDSolver.h"

/** @addtogroup fel_fields
 * @{ */

/** @brief Uniform Cartesian mesh with double-precision physical coordinates.
 * @tparam Dim Number of mesh dimensions; the FEL application uses three.
 * Coordinate units come from the caller; FEL uses its internal length units.
 */
template <unsigned Dim>
using Mesh_t = ippl::UniformCartesian<double, Dim>;

/** @brief Particle spatial layout tied to the same Cartesian mesh as the fields.
 * @tparam T Particle-coordinate scalar type.
 * @tparam Dim Spatial dimension.
 */
template <typename T, unsigned Dim>
using PLayout_t = typename ippl::ParticleSpatialLayout<T, Dim, Mesh_t<Dim>>;

/** @brief Cell centering inherited from UniformCartesian::DefaultCentering.
 * @tparam Dim Spatial dimension of the mesh.
 * All components of each vector field share this centering.
 */
template <unsigned Dim>
using Centering_t = typename Mesh_t<Dim>::DefaultCentering;

/** @brief Distributed index-domain layout used to partition a mesh across ranks.
 * @tparam Dim Number of index dimensions.
 */
template <unsigned Dim>
using FieldLayout_t = ippl::FieldLayout<Dim>;

using size_type = ippl::detail::size_type;  ///< IPPL's common size/index-count type.

/** @brief Fixed-size IPPL vector with the requested component type and count.
 * @tparam T Scalar component type.
 * @tparam Dim Number of components, not necessarily the mesh dimension.
 */
template <typename T, unsigned Dim>
using Vector_t = ippl::Vector<T, Dim>;

/** @brief Cell-centred distributed field on the FEL Cartesian mesh type.
 * @tparam T Value stored at each cell; may itself be a vector.
 * @tparam Dim Mesh dimension.
 * @tparam ViewArgs Additional Kokkos view properties forwarded to ippl::Field.
 * Memory space, allocation and halo operations are handled by IPPL; this alias
 * by itself neither allocates data nor performs a host/device transfer.
 */
template <typename T, unsigned Dim, class... ViewArgs>
using Field = ippl::Field<T, Dim, Mesh_t<Dim>, Centering_t<Dim>, ViewArgs...>;

/** @brief Three-component collocated electric or magnetic field.
 * @tparam T Scalar component type.
 * @tparam Dim Mesh dimension; the vector still has exactly three components.
 * @tparam ViewArgs Additional Kokkos view properties forwarded to Field.
 */
template <typename T, unsigned Dim, class... ViewArgs>
using VField_t = Field<Vector_t<T, 3>, Dim, ViewArgs...>;

/** @brief Collocated source-field storage, also suitable for the four-potential.
 * @tparam T Scalar component type.
 * @tparam Dim Mesh dimension; each value has Dim+1 components.
 * @tparam ViewArgs Additional Kokkos view properties forwarded to Field.
 *
 * For the three-dimensional FEL source, component 0 is charge density rho and
 * components 1..3 are current density (Jx,Jy,Jz), in the internal units with
 * dimensionless light speed one. The solver uses the same storage type for
 * (phi,Ax,Ay,Az) at three time levels. This alias describes storage, not a
 * deposition algorithm or a material constitutive relation.
 */
template <typename T, unsigned Dim, class... ViewArgs>
using SourceField_t = Field<Vector_t<T, Dim + 1>, Dim, ViewArgs...>;

/** @brief Active MITHRA-style nonstandard vacuum FDTD solver with Mur boundaries.
 * @tparam T Scalar component type.
 * @tparam Dim Mesh dimension; the current solver implementation is used in 3D.
 *
 * NonStandardFDTDSolver advances the four-potential, then reconstructs
 * @f$\mathbf E=-\partial_t\mathbf A-\nabla\phi@f$ and
 * @f$\mathbf B=\nabla\times\mathbf A@f$. Its active absorbing boundary policy
 * applies second-order Mur conditions to the potentials. These approximate an
 * open boundary; they are not a perfectly matched layer or a reflection guarantee.
 *
 * In internal units the active solver sets @f$\Delta t=h_z@f$ and requires
 * @f$(h_z/h_x)^2+(h_z/h_y)^2<1@f$. The standard solver alternative below is
 * commented out: changing the active solver is a source-code choice, not JSON input.
 */
template <typename T, unsigned Dim>
using FDTDSolver_t =
    ippl::NonStandardFDTDSolver<VField_t<T, Dim>, SourceField_t<T, Dim>, ippl::absorbing>;

// using FDTDSolver_t =
//     ippl::StandardFDTDSolver<VField_t<T, Dim>, SourceField_t<T, Dim>, ippl::absorbing>;

/** @brief Cast every vector component on the host or in a Kokkos device kernel.
 * @tparam U Destination scalar type.
 * @tparam T Source scalar type.
 * @tparam D Number of vector components.
 * @param v Input vector.
 * @return A value vector whose component k is static_cast<U>(v[k]).
 * No range checking or allocation is performed; normal C++ cast rules apply.
 */
template <typename U, typename T, unsigned D>
KOKKOS_INLINE_FUNCTION ippl::Vector<U, D> vector_cast(const ippl::Vector<T, D>& v) {
    ippl::Vector<U, D> ret;
    for (unsigned k = 0; k < D; ++k) {
        ret[k] = static_cast<U>(v[k]);
    }
    return ret;
}

/** @} */

#endif
