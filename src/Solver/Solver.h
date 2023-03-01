//
// Class Solver
//   Base class for all solvers.
//
// Copyright (c) 2021, Matthias Frey, University of St Andrews, St Andrews, Scotland
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

#ifndef IPPL_SOLVER_H
#define IPPL_SOLVER_H

#include "Field/Field.h"
#include "Utility/ParameterList.h"

namespace ippl {

    template <
        typename Tlhs, typename Trhs, unsigned Dim, class M = UniformCartesian<double, Dim>,
        class C = typename M::DefaultCentering>
    class Solver {
    public:
        using lhs_type = Field<Tlhs, Dim, M, C>;
        using rhs_type = Field<Trhs, Dim, M, C>;

        /*!
         * Default constructor
         */
        Solver() {}

        /*!
         * Convenience constructor with LHS and RHS parameters
         * @param lhs The LHS for the problem to solve
         * @param rhs The RHS for the problem to solve
         */
        Solver(lhs_type& lhs, rhs_type& rhs) {
            setLhs(lhs);
            setRhs(rhs);
        }

        /*!
         * Update one of the solver's parameters
         * @param key The parameter key
         * @param value The new value
         * @throw IpplException Fails if there is no existing parameter with the given key
         */
        template <typename T>
        void updateParameter(const std::string& key, const T& value) {
            params_m.update<T>(key, value);
        }

        /*!
         * Updates all solver parameters based on values in another parameter set
         * @param params Parameter list with updated values
         * @throw IpplException Fails if the provided parameter list includes keys not already
         * present
         */
        void updateParameters(const ParameterList& params) { params_m.update(params); }

        /*!
         * Merges another parameter set into the solver's parameters, overwriting
         * existing parameters in case of conflict
         * @param params Parameter list with desired values
         */
        void mergeParameters(const ParameterList& params) { params_m.merge(params); }

        /*!
         * Set the problem LHS
         * @param lhs Reference to problem LHS field
         */
        void setLhs(lhs_type& lhs) { lhs_mp = &lhs; }

        /*!
         * Set the problem RHS
         * @param rhs Reference to problem RHS field
         */
        virtual void setRhs(rhs_type& rhs) { rhs_mp = &rhs; }

    protected:
        ParameterList params_m;

        rhs_type* rhs_mp;
        lhs_type* lhs_mp;

        /*!
         * Utility function for initializing a solver's default
         * parameters (to be overridden for each base class)
         */
        virtual void setDefaultParameters() {}
    };
}  // namespace ippl

#endif
