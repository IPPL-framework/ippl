// Class FEMPoissonSolver
//   Solves the poisson equation using finite element methods and Conjugate
//   Gradient

#ifndef IPPL_FEMPOISSONSOLVER_H
#define IPPL_FEMPOISSONSOLVER_H

// #include "FEM/FiniteElementSpace.h"
#include "LaplaceHelpers.h"
#include "LinearSolvers/PCG.h"
#include "Poisson.h"

namespace ippl {

    /**
     * @brief A solver for the poisson equation using finite element methods and
     * Conjugate Gradient (CG)
     *
     * @tparam FieldLHS field type for the left hand side
     * @tparam FieldRHS field type for the right hand side
     */
    template <typename FieldLHS, typename FieldRHS = FieldLHS>
    class FEMPoissonSolver : public Poisson<FieldLHS, FieldRHS> {
        constexpr static unsigned Dim = FieldLHS::dim;
        using Tlhs                    = typename FieldLHS::value_type;

    public:
        using Base = Poisson<FieldLHS, FieldRHS>;
        using typename Base::lhs_type, typename Base::rhs_type;
        using OperatorRet        = UnaryMinus<detail::meta_laplace<lhs_type>>;
        using LowerRet           = UnaryMinus<detail::meta_lower_laplace<lhs_type>>;
        using UpperRet           = UnaryMinus<detail::meta_upper_laplace<lhs_type>>;
        using UpperAndLowerRet   = UnaryMinus<detail::meta_upper_and_lower_laplace<lhs_type>>;
        using InverseDiagonalRet = lhs_type;

        // PCG (Preconditioned Conjugate Gradient) is the solver algorithm used
        using PCGSolverAlgorithm_t = PCG<OperatorRet, LowerRet, UpperRet, UpperAndLowerRet,
                                         InverseDiagonalRet, FieldLHS, FieldRHS>;

        // FEM Space types
        using ElementType =
            std::conditional_t<Dim == 1, ippl::EdgeElement<Tlhs>,
                               std::conditional_t<Dim == 2, ippl::QuadrilateralElement<Tlhs>,
                                                  ippl::HexahedralElement<Tlhs>>>;

        using QuadratureType = GaussJacobiQuadrature<Tlhs, 5, ElementType>;

        FEMPoissonSolver(lhs_type& lhs, rhs_type& rhs,
                         const std::function<Tlhs(const Vector<Tlhs, Dim>&)>& rhs_f)
            : Base(lhs, rhs)
            , refElement_m()
            , quadrature_m(refElement_m, 0.0, 0.0)
            , lagrangeSpace_m(lhs.get_mesh(), refElement_m, quadrature_m) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();

            lagrangeSpace_m.evaluateLoadVector(rhs, rhs_f);
        }

        /**
         * @brief Solve the poisson equation using finite element methods.
         * The problem is described by -laplace(lhs) = rhs
         */
        void solve() override {
            const Vector<std::size_t, Dim> zeroNdIndex = Vector<std::size_t, Dim>(0);

            // We can pass the zeroNdIndex here, since the transformation jacobian does not depend
            // on translation
            const auto firstElementVertexPoints =
                lagrangeSpace_m.getElementMeshVertexPoints(zeroNdIndex);

            // Compute Inverse Transpose Transformation Jacobian ()
            const Vector<Tlhs, Dim> DPhiInvT =
                refElement_m.getInverseTransposeTransformationJacobian(firstElementVertexPoints);

            // Compute absolute value of the determinante of the transformation jacobian (|det D
            // Phi_K|)
            const Tlhs absDetDPhi = std::abs(
                refElement_m.getDeterminantOfTransformationJacobian(firstElementVertexPoints));

            const int num_element_dofs = this->langrangeSpace_m.numElementDOFs;

            const auto poissonEquationEval =
                [DPhiInvT, absDetDPhi](
                    const std::size_t& i, const std::size_t& j,
                    const Vector<Vector<Tlhs, Dim>, num_element_dofs>& grad_b_q_k) {
                    return dot((DPhiInvT * grad_b_q_k[j]), (DPhiInvT * grad_b_q_k[i])).apply()
                           * absDetDPhi;
                };

            const auto algoOperator = [poissonEquationEval,
                                       this](const lhs_type& field) -> lhs_type {
                return lagrangeSpace_m.evaluateAx(field, poissonEquationEval);
            };

            pcg_algo_m.setOperator(algoOperator);
            pcg_algo_m(*(this->lhs_mp), *(this->rhs_mp), this->params_m);

            int output = this->params_m.template get<int>("output_type");
            if (output & Base::GRAD) {
                *(this->grad_mp) = -grad(*(this->lhs_mp));
            }
        }

        /**
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        int getIterationCount() { return pcg_algo_m.getIterationCount(); }

        /**
         * Query the residue
         * @return Residue norm from last solve
         */
        Tlhs getResidue() const { return pcg_algo_m.getResidue(); }

    protected:
        PCGSolverAlgorithm_t pcg_algo_m;

        virtual void setDefaultParameters() override {
            this->params_m.add("max_iterations", 1000);
            this->params_m.add("tolerance", (Tlhs)1e-13);
        }

        ElementType refElement_m;
        QuadratureType quadrature_m;
        LagrangeSpace<Tlhs, Dim, 1, QuadratureType, FieldLHS, FieldRHS> lagrangeSpace_m;
    };

}  // namespace ippl

#endif
