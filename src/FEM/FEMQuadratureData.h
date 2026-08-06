#ifndef IPPL_FEM_QUADRATURE_DATA_H
#define IPPL_FEM_QUADRATURE_DATA_H

#include "Types/Vector.h"

namespace ippl {

/**
 * @brief Per-quadrature-node basis data passed to evaluateAx evaluator functors.
 *
 * @tparam TVal Type of basis values at each local DOF (scalar T for Lagrange,
 *              Vector<T, Dim> for Nedelec).
 * @tparam TDeriv Type of the spatial derivative data at each local DOF
 *                (gradient for Lagrange, curl for Nedelec).
 * @tparam numElementDOFs Number of local DOFs per element.
 */
template <typename TVal, typename TDeriv, unsigned numElementDOFs>
struct QuadratureData {
    const Vector<TVal, numElementDOFs>& val_q;
    const Vector<TDeriv, numElementDOFs>& deriv_q;
};

}  // namespace ippl

#endif
