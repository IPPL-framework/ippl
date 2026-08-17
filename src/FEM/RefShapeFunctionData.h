#ifndef IPPL_FEM_REF_SHAPE_FUNCTION_DATA_H
#define IPPL_FEM_REF_SHAPE_FUNCTION_DATA_H

#include "Types/Vector.h"

namespace ippl {

/**
 * @brief Reference-element shape-function data at a single quadrature node.
 *
 * Bundles, for one quadrature node q, the per-local-DOF shape-function values (@ref val_q) and
 * their derivatives (@ref deriv_q) -- a gradient for Lagrange, a curl for Nedelec. Both members
 * are reference-element quantities: the values and the derivatives with respect to the reference
 * coordinates. The physical mapping (Jacobian / Piola transform, |det DPhi|) is applied by the
 * evaluator functor that consumes this struct, not stored here. The members are held by const
 * reference, so the struct is a lightweight, non-owning view constructed on the fly during
 * element-matrix assembly.
 *
 * The @c _q suffix denotes "evaluated at quadrature node q" (not a component index).
 *
 * @tparam TVal Type of the shape-function value at each local DOF (scalar T for Lagrange,
 *              Vector<T, Dim> for Nedelec).
 * @tparam TDeriv Type of the reference derivative at each local DOF
 *                (gradient for Lagrange, curl for Nedelec).
 * @tparam numElementDOFs Number of local DOFs per element.
 */
template <typename TVal, typename TDeriv, unsigned numElementDOFs>
struct RefShapeFunctionData {
    const Vector<TVal, numElementDOFs>& val_q;
    const Vector<TDeriv, numElementDOFs>& deriv_q;
};

}  // namespace ippl

#endif
