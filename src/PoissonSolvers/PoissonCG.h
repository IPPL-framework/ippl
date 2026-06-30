//
// Class PoissonCG
//   Solves the Poisson problem with the CG algorithm
//

#ifndef IPPL_POISSON_CG_H
#define IPPL_POISSON_CG_H

#include "LaplaceHelpers.h"
#include "LinearSolvers/PCG.h"
#include "LinearSolvers/PreconditionerValidation.h"
#include "Poisson.h"
namespace ippl {

    // IPPL_SOLVER_OPERATOR_WRAPPER is defined once in LinearSolvers/Preconditioner.h
    // (re-exported through this header via PCG.h). Defining it again here used to
    // silently shadow that definition with a by-value lambda, which copies the
    // Field on every op_m() call and reintroduces the per-iteration cudaMalloc
    // in the halo exchange (the realloc never propagates back to the original
    // Field). Keep a single by-reference definition.

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
            setSolver(*(this->lhs_mp));
        }

        void setLhs(lhs_type& lhs) override {
            Base::setLhs(lhs);
            setSolver(lhs);
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
                // Get the preconditioner type,
                // if it is not part of the valid list of preconditioners, throw an error.
                std::string preconditioner_type =
                    this->params_m.template get<std::string>("preconditioner_type");
                preconditioner_validation::throwIfUnknownType(preconditioner_type,
                                                              "PoissonCG::setSolver");

                // Read in the preconditioner parameters
                int level    = this->params_m.template get<int>("newton_level");
                int degree   = this->params_m.template get<int>("chebyshev_degree");
                int inner    = this->params_m.template get<int>("gauss_seidel_inner_iterations");
                int outer    = this->params_m.template get<int>("gauss_seidel_outer_iterations");
                double omega = this->params_m.template get<double>("ssor_omega");
                int richardson_iterations =
                    this->params_m.template get<int>("richardson_iterations");
                int communication = this->params_m.template get<int>("communication");

                // Extract Multigrid params
                int mg_pre = this->params_m.template get<int>(
                    "mg_pre_smooth_iters", pcg_preconditioner_defaults::mg_pre_smooth);
                int mg_post = this->params_m.template get<int>(
                    "mg_post_smooth_iters", pcg_preconditioner_defaults::mg_post_smooth);
                double mg_omega = this->params_m.template get<double>(
                    "mg_omega", pcg_preconditioner_defaults::mg_omega);
                unsigned mg_min_cells = static_cast<unsigned>(this->params_m.template get<int>(
                    "min_cells_per_rank_per_dim",
                    static_cast<int>(pcg_preconditioner_defaults::mg_min_cells)));
                bool mg_communication = communication;

                Inform warn("PoissonCG");
                // After reading in preconditioner parameters, if they are invalid,
                // the user is warned that the parameter is invalid, and a default
                // parameter is used.
                preconditioner_validation::sanitizeParams(
                    preconditioner_type, warn, level, degree, richardson_iterations, inner, outer,
                    omega, &communication, mg_pre, mg_post, mg_omega, mg_min_cells);
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
                        omega, mg_pre, mg_post, mg_omega, mg_min_cells, mg_communication);
                } else {
                    algo_m->setPreconditioner(
                        IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-lower_laplace_no_comm, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-upper_laplace_no_comm, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-upper_and_lower_laplace_no_comm, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(negative_inverse_diagonal_laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(diagonal_laplace, lhs_type), alpha, beta,
                        preconditioner_type, level, degree, richardson_iterations, inner, outer,
                        omega, mg_pre, mg_post, mg_omega, mg_min_cells, mg_communication);
                }
            } else {
                algo_m = std::move(
                    std::make_unique<CG<OperatorRet, LowerRet, UpperRet, UpperAndLowerRet,
                                        InverseDiagonalRet, DiagRet, FieldLHS, FieldRHS>>());
            }
            algo_m->initializeFields(lhs.get_mesh(), lhs.getLayout());
        }

        void solve() override {
            // \todo TODO add a check for mesh changes for alpha and beta for preconditioners

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
