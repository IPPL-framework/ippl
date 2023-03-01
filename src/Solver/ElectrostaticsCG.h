//
// Class ElectrostaticsCG
//   Solves electrostatics problems with the CG algorithm
//
// Copyright (c) 2021
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

#ifndef IPPL_ELECTROSTATICS_CG_H
#define IPPL_ELECTROSTATICS_CG_H

#include "Electrostatics.h"
#include "PCG.h"

namespace ippl {

// Expands to a lambda that acts as a wrapper for a differential operator
// fun: the function for which to create the wrapper, such as ippl::laplace
// type: the argument type, which should match the LHS type for the solver
#define IPPL_SOLVER_OPERATOR_WRAPPER(fun, type) \
    [](type arg) {                              \
        return fun(arg);                        \
    }

    template <
        typename Tlhs, typename Trhs, unsigned Dim, class M = UniformCartesian<double, Dim>,
        class C = typename M::DefaultCentering>
    class ElectrostaticsCG : public Electrostatics<Tlhs, Trhs, Dim, M, C> {
    public:
        using lhs_type = typename Solver<Tlhs, Trhs, Dim, M, C>::lhs_type;
        using rhs_type = typename Solver<Tlhs, Trhs, Dim, M, C>::rhs_type;
        using OpRet    = UnaryMinus<detail::meta_laplace<lhs_type>>;
        using algo     = PCG<Tlhs, Trhs, Dim, OpRet, M, C>;
        using Base     = Electrostatics<Tlhs, Trhs, Dim, M, C>;

        ElectrostaticsCG() : Base() {
            setDefaultParameters();
        }

        ElectrostaticsCG(lhs_type& lhs, rhs_type& rhs) : Base(lhs, rhs) {
            setDefaultParameters();
        }

        void solve() override {
            algo_m.setOperator(IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, lhs_type));
            algo_m(*(this->lhs_mp), *(this->rhs_mp), this->params_m);

            int output = this->params_m.template get<int>("output_type");
            if (output & Base::GRAD) {
                *(this->grad_mp) = grad(*(this->lhs_mp));
            }
        }

        /*!
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        int getIterationCount() {
            return algo_m.getIterationCount();
        }

    protected:
        algo algo_m = algo();

        virtual void setDefaultParameters() override {
            this->params_m.add("max_iterations", 1000);
            this->params_m.add("tolerance", (Tlhs)1e-13);
        }
    };

}  // namespace ippl

#endif
