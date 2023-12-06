//
// Class PoissonMG
//   Solves the Poisson problem with a Multigrid solver
//

#ifndef IPPL_POISSON_MG_H
#define IPPL_POISSON_MG_H

#include "LinearSolvers/PCG.h"
#include "LinearSolvers/Multigrid.h"
#include "Poisson.h"

namespace ippl {

// Expands to a lambda that acts as a wrapper for a differential operator
// fun: the function for which to create the wrapper, such as ippl::laplace
// type: the argument type, which should match the LHS type for the solver
#define IPPL_SOLVER_OPERATOR_WRAPPER(fun, type) \
    [](type arg) {                              \
        return fun(arg);                        \
    }

    template <typename FieldLHS, typename FieldRHS = FieldLHS>
    class PoissonMG : public Poisson<FieldLHS, FieldRHS> {
        using Tlhs = typename FieldLHS::value_type;

    public:
        using Base = Poisson<FieldLHS, FieldRHS>;
        using typename Base::lhs_type, typename Base::rhs_type;
        using OpRet = UnaryMinus<detail::meta_laplace<lhs_type>>;
        using algo = Multigrid<OpRet, FieldLHS, FieldRHS>;

        PoissonMG()
                : Base() {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        PoissonMG(lhs_type& lhs, rhs_type& rhs)
                : Base(lhs, rhs) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        void solve() override {
            algo_m.setOperator(IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, lhs_type));
            ParameterList CGParameters;
            CGParameters.add("max_iterations", 100);
            CGParameters.add("tolerance" , 1e-5);
            algo_m.setCG(CGParameters , IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, lhs_type));
            algo_m(*(this->lhs_mp), *(this->rhs_mp), this->params_m);
            int output = this->params_m.template get<int>("output_type");
            if (output & Base::GRAD) {
                *(this->grad_mp) = -grad(*(this->lhs_mp));
            }
        }

        /*!
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        int getIterationCount() { return algo_m.getIterationCount(); }

        /*!
         * Query the residue
         * @return Residue norm from last solve
         */
        Tlhs getResidue() const { return algo_m.getResidue(); }

    protected:
        algo algo_m = algo();

        void setDefaultParameters() override {
            this->params_m.add("max_iterations", 2000);
            this->params_m.add("levels", 1);
            this->params_m.add("tolerance", (Tlhs)1e-13);
        }
    };

}  // namespace ippl

#endif // IPPL_POISSON_MG_H
