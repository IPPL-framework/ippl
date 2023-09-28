// Class FEMPoissonSolver
//   Solves the poisson equation using finite element methods and Conjugate
//   Gradient

#ifndef IPPL_FEMPOISSONSOLVER_H
#define IPPL_FEMPOISSONSOLVER_H

#include "PCG.h"
#include "Solver/Solver.h"

namespace ippl {

/**
 * @brief A solver for the poisson equation using finite element methods and
 * Conjugate Gradient (CG)
 *
 * @tparam FieldLHS field type for the left hand side
 * @tparam FieldRHS field type for the right hand side
 */
template <typename FieldLHS, typename FieldRHS = FieldLHS>
class FEMPoissonSolver : public Solver<FieldLHS, FieldRHS> {
    constexpr static unsigned Dim = FieldLHS::dim;
    using Tlhs                    = typename FieldLHS::value_type;
    using Trhs                    = typename FieldRHS::value_type;

   public:
    using Base = Solver<FieldLHS, FieldRHS>;
    using typename Base::lhs_type, typename Base::rhs_type;
    using algo =
        PCG<UnaryMinus<detail::meta_laplace<lhs_type>>, FieldLHS, FieldRHS>;

    /**
     * @brief Default constructor for FEM poisson solver
     */
    FEMPoissonSolver() : Base() {
        static_assert(std::is_floating_point<Trhs>::value,
                      "Not a floating point type");
        setDefaultParameters();
    }

    /**
     * @brief Construct a new FEMPoissonSolver object
     *
     * @param lhs left hand side: -laplace(lhs)
     * @param rhs right hand side
     */
    FEMPoissonSolver(lhs_type& lhs, rhs_type& rhs) : Base(lhs, rhs) {
        static_assert(std::is_floating_point<Trhs>::value,
                      "Not a floating point type");
        setDefaultParameters();
    }

    /**
     * @brief Solve the poisson equation using finite element methods.
     * The problem is described by -laplace(lhs) = rhs
     */
    void solve() {
        algo_m.setOperator(IPPL_SOVLER_OPERATOR_WRAPPER(-laplace, lhs_type));
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

    virtual void setDefaultParameters() override {
        this->params_m.add("max_iterations", 1000);
        this->params_m.add("tolerance", (Tlhs)1e-13);
    }
};

}  // namespace ippl

#endif