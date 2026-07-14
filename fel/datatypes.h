#ifndef IPPL_FEL_DATATYPES_H
#define IPPL_FEL_DATATYPES_H

#include "MaxwellSolvers/StandardFDTDSolver.h"
#include "MaxwellSolvers/NonStandardFDTDSolver.h"

// Type aliases for the FEL module, mirroring alpine/datatypes.h.
//
// The FEL simulation solves Maxwell's equations with an FDTD solver coupled to
// a relativistic electron bunch. Two field flavours are needed:
//   * VField_t      : E and B, three-component vector fields.
//   * SourceField_t : the four-current source (rho, Jx, Jy, Jz)

template <unsigned Dim>
using Mesh_t = ippl::UniformCartesian<double, Dim>;

template <typename T, unsigned Dim>
using PLayout_t = typename ippl::ParticleSpatialLayout<T, Dim, Mesh_t<Dim>>;

template <unsigned Dim>
using Centering_t = typename Mesh_t<Dim>::DefaultCentering;

template <unsigned Dim>
using FieldLayout_t = ippl::FieldLayout<Dim>;

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim>
using Vector_t = ippl::Vector<T, Dim>;

template <typename T, unsigned Dim, class... ViewArgs>
using Field = ippl::Field<T, Dim, Mesh_t<Dim>, Centering_t<Dim>, ViewArgs...>;

// Electric and magnetic fields: three-component vector field on the mesh.
template <typename T, unsigned Dim, class... ViewArgs>
using VField_t = Field<Vector_t<T, 3>, Dim, ViewArgs...>;

// Four-current source field: [0] = charge density
// [1..Dim] = current density.
template <typename T, unsigned Dim, class... ViewArgs>
using SourceField_t = Field<Vector_t<T, Dim + 1>, Dim, ViewArgs...>;

// The Maxwell FDTD solver used by the FEL simulation. Boundary conditions are
// absorbing (second-order Mur) so radiation leaves the domain cleanly.
//
// Both standard finite difference and the mithra-style non-standard finite difference 
// are available.
template <typename T, unsigned Dim>
using FDTDSolver_t =
    ippl::NonStandardFDTDSolver<VField_t<T, Dim>, SourceField_t<T, Dim>, ippl::absorbing>;

//using FDTDSolver_t =
//    ippl::StandardFDTDSolver<VField_t<T, Dim>, SourceField_t<T, Dim>, ippl::absorbing>;

// Component-wise cast of a Vector to another scalar type.
template <typename U, typename T, unsigned D>
KOKKOS_INLINE_FUNCTION ippl::Vector<U, D> vector_cast(const ippl::Vector<T, D>& v) {
    ippl::Vector<U, D> ret;
    for (unsigned k = 0; k < D; ++k) {
        ret[k] = static_cast<U>(v[k]);
    }
    return ret;
}

#endif
