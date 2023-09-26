//
// Class SolverAlgorithm
//   Base class for solver algorithms
//

#ifndef IPPL_SOLVER_ALGORITHM_H
#define IPPL_SOLVER_ALGORITHM_H

#include <functional>

#include "Utility/ParameterList.h"

namespace ippl {

    template <typename FieldLHS, typename FieldRHS>
    class SolverAlgorithm {
    public:
        using lhs_type = FieldLHS;
        using rhs_type = FieldRHS;

        /*!
         * Solve the problem described by Op(lhs) = rhs, where Op is an unspecified
         * differential operator (handled by derived classes)
         * @param lhs The problem's LHS
         * @param rhs The problem's RHS
         * @param params A set of parameters for the solver algorithm
         */
        virtual void operator()(lhs_type& lhs, rhs_type& rhs, const ParameterList& params) = 0;
    };

}  // namespace ippl

#endif
