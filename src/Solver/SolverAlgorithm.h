//
// Class SolverAlgorithm
//   Base class for solver algorithms
//
// Copyright (c) 2021 Alessandro Vinciguerra, ETH Zürich, Zurich, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

#ifndef IPPL_SOLVER_ALGORITHM_H
#define IPPL_SOLVER_ALGORITHM_H

#include <functional>
#include "Utility/ParameterList.h"

namespace ippl {

    template <typename Tlhs, typename Trhs, unsigned Dim, class Mesh, class Centering>
    class SolverAlgorithm {
    public:
        using lhs_type = Field<Tlhs, Dim, Mesh, Centering>;
        using rhs_type = Field<Trhs, Dim, Mesh, Centering>;

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
