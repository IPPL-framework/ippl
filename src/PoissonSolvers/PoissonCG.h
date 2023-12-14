//
// Class PoissonCG
//   Solves the Poisson problem with the CG algorithm
//

#ifndef IPPL_POISSON_CG_H
#define IPPL_POISSON_CG_H

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
    class PoissonCG : public Poisson<FieldLHS, FieldRHS> {
        using Tlhs = typename FieldLHS::value_type;

    public:
        using Base = Poisson<FieldLHS, FieldRHS>;
        using typename Base::lhs_type, typename Base::rhs_type;
        using OpRet = UnaryMinus<detail::meta_laplace<lhs_type>>;
        using PreRet = lhs_type;

        PoissonCG()
            : Base() , algo_m(nullptr) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();

        }

        PoissonCG(lhs_type& lhs, rhs_type& rhs)
            : Base(lhs, rhs) , algo_m(nullptr) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        void setSolver() {
            std::string solver_type = this->params_m.template get<std::string>("solver");
            if (solver_type == "preconditioned") {
                algo_m = std::move(std::make_unique<PCG<OpRet, PreRet, FieldLHS, FieldRHS>>());
                std::string preconditioner_type = this->params_m.template get<std::string>("preconditioner_type");
                int level = this->params_m.template get<int>("newton_level");
                int degree = this->params_m.template get<int>("chebyshev_degree");
                int inner = this->params_m.template get<int>("gauss_seidel_inner_iterations");
                int outer = this->params_m.template get<int>("gauss_seidel_outer_iterations");
                int richardson_iterations = this->params_m.template get<int>("richardson_iterations");
                algo_m->setPreconditioner(preconditioner_type, level, degree, richardson_iterations, inner, outer);

            } else {
                algo_m = std::move(std::make_unique<CG<OpRet, PreRet, FieldLHS, FieldRHS>>());
            }
        }

        void solve() override {
            setSolver();
            algo_m->setOperator(IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, lhs_type));
            algo_m->operator()(*(this->lhs_mp), *(this->rhs_mp), this->params_m);

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
        int getIterationCount() { return algo_m->getIterationCount(); }

        /*!
         * Query the residue
         * @return Residue norm from last solve
         */
        Tlhs getResidue() const { return algo_m->getResidue(); }

    protected:
        std::unique_ptr<CG<OpRet, PreRet, FieldLHS, FieldRHS>> algo_m;

        void setDefaultParameters() override {
            this->params_m.add("max_iterations", 2000);
            this->params_m.add("tolerance", (Tlhs)1e-13);
        }
    };

}  // namespace ippl

#endif
