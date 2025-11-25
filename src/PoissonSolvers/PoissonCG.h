//
// Class PoissonCG
//   Solves the Poisson problem with the CG algorithm
//

#ifndef IPPL_POISSON_CG_H
#define IPPL_POISSON_CG_H

#include "LaplaceHelpers.h"
#include "LinearSolvers/PCG.h"
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
        using Base                    = Poisson<FieldLHS, FieldRHS>;
        constexpr static unsigned Dim = FieldLHS::dim;
        using typename Base::lhs_type, typename Base::rhs_type;
        using OperatorRet        = UnaryMinus<detail::meta_laplace<lhs_type>>;
        using LowerRet           = UnaryMinus<detail::meta_lower_laplace<lhs_type>>;
        using UpperRet           = UnaryMinus<detail::meta_upper_laplace<lhs_type>>;
        using UpperAndLowerRet   = UnaryMinus<detail::meta_upper_and_lower_laplace<lhs_type>>;
        using InverseDiagonalRet = double;
        using DiagRet            = double;

        PoissonCG()
            : Base()
            , algo_m(nullptr) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        PoissonCG(lhs_type& lhs, rhs_type& rhs)
            : Base(lhs, rhs)
            , algo_m(nullptr) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();
            algo_m->initializeFields(rhs.get_mesh(), rhs.getLayout());
        }

        void setRhs(rhs_type& rhs) override {
            Base::setRhs(rhs);
            algo_m->initializeFields(rhs.get_mesh(), rhs.getLayout());
        }

        void setSolver(lhs_type lhs) {
            std::string solver_type            = this->params_m.template get<std::string>("solver");
            typename lhs_type::Mesh_t mesh     = lhs.get_mesh();
            typename lhs_type::Layout_t layout = lhs.getLayout();
            double beta                        = 0;
            double alpha                       = 0;
            if (solver_type == "preconditioned") {
                algo_m = std::move(
                    std::make_unique<PCG<OperatorRet, LowerRet, UpperRet, UpperAndLowerRet,
                                         InverseDiagonalRet, DiagRet, FieldLHS, FieldRHS>>());
                std::string preconditioner_type =
                    this->params_m.template get<std::string>("preconditioner_type");
                int level    = this->params_m.template get<int>("newton_level");
                int degree   = this->params_m.template get<int>("chebyshev_degree");
                int inner    = this->params_m.template get<int>("gauss_seidel_inner_iterations");
                int outer    = this->params_m.template get<int>("gauss_seidel_outer_iterations");
                double omega = this->params_m.template get<double>("ssor_omega");
                int richardson_iterations =
                    this->params_m.template get<int>("richardson_iterations");
                int communication = this->params_m.template get<int>("communication");
                // Analytical eigenvalues for the d dimensional laplace operator
                // Going brute force through all possible eigenvalues seems to be the only way to
                // find max and min

                unsigned long n;
                double h;
                for (unsigned int d = 0; d < Dim; ++d) {
                    n                = mesh.getGridsize(d);
                    h                = mesh.getMeshSpacing(d);
                    double local_min = 4 / std::pow(h, 2);  // theoretical maximum
                    double local_max = 0;
                    double test;
                    for (unsigned int i = 1; i < n; ++i) {
                        test = 4. / std::pow(h, 2) * std::pow(std::sin(i * M_PI * h / 2.), 2);
                        if (test > local_max) {
                            local_max = test;
                        }
                        if (test < local_min) {
                            local_min = test;
                        }
                    }
                    beta += local_max;
                    alpha += local_min;
                }
                if (communication) {
                    algo_m->setPreconditioner(
                        IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-lower_laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-upper_laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-upper_and_lower_laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(negative_inverse_diagonal_laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(diagonal_laplace, lhs_type), alpha, beta,
                        preconditioner_type, level, degree, richardson_iterations, inner, outer,
                        omega);
                } else {
                    algo_m->setPreconditioner(
                        IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-lower_laplace_no_comm, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-upper_laplace_no_comm, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-upper_and_lower_laplace_no_comm, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(negative_inverse_diagonal_laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(diagonal_laplace, lhs_type), alpha, beta,
                        preconditioner_type, level, degree, richardson_iterations, inner, outer,
                        omega);
                }
            } else {
                algo_m = std::move(
                    std::make_unique<CG<OperatorRet, LowerRet, UpperRet, UpperAndLowerRet,
                                        InverseDiagonalRet, DiagRet, FieldLHS, FieldRHS>>());
            }
        }

        void solve() override {
            setSolver(*(this->lhs_mp));
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
        std::unique_ptr<CG<OperatorRet, LowerRet, UpperRet, UpperAndLowerRet, InverseDiagonalRet,
                           DiagRet, FieldLHS, FieldRHS>>
            algo_m;

        void setDefaultParameters() override {
            this->params_m.add("max_iterations", 2000);
            this->params_m.add("tolerance", (Tlhs)1e-13);
            this->params_m.add("solver", "non-preconditioned");
        }
    };

}  // namespace ippl

#endif
