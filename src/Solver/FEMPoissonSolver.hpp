// Class FEMPoissonSolver
//   A finite-element Poisson solver.

#ifndef IPPL_FEMPOISSONSOLVER_H
#define IPPL_FEMPOISSONSOLVER_H

#include "Solver/Solver.h"

namespace ippl {

template <typename FieldLHS, typename FieldRHS>
class FEMPoissonSolver : public Solver<FieldLHS, FieldRHS> {};

}  // namespace ippl

#endif