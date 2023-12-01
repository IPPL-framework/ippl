// Class FEMPoissonSolver
//   Solves the poisson equation using finite element methods and Conjugate
//   Gradient

#ifndef IPPL_FEMPOISSONSOLVER_H
#define IPPL_FEMPOISSONSOLVER_H

// #include "FEM/FiniteElementSpace.h"
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

        // PCG (Preconditioned Conjugate Gradient) is the solver algorithm used
        using OpRet = UnaryMinus<detail::meta_laplace<lhs_type>>;
        using algo  = PCG<OpRet, FieldLHS, FieldRHS>;

        // FEM Space types
        using ElementType    = HexahedralElement<Tlhs>;  // 3D Element
        using QuadratureType = GaussJacobiQuadrature<Tlhs, 5, ElementType>;

        // FEMPoissonSolver()
        //     : Base(), ref_element_m(), quadrature_m() {
        //     static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
        //     setDefaultParameters();
        // }

        FEMPoissonSolver(lhs_type& lhs, rhs_type& rhs)
            : Base(lhs, rhs)
            , refElement_m()
            , quadrature_m()
            , lagrangeSpace_m(lhs.get_mesh(), refElement_m, quadrature_m) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        /**
         * @brief Solve the poisson equation using finite element methods.
         * The problem is described by -laplace(lhs) = rhs
         */
        void solve() override {
            const std::size_t Dim            = 3;
            const std::size_t NumElementDOFs = 8;

            const Vector<std::size_t, Dim> zeroNdIndex = Vector<std::size_t, Dim>(0);

            // Inverse Transpose Transformation Jacobian
            const Vector<Tlhs, Dim> DPhiInvT =
                this->ref_element_m.getInverseTransposeTransformationJacobian(
                    this->getElementMeshVertexIndices(zeroNdIndex));

            // Absolute value of det Phi_K
            const Tlhs absDetDPhi =
                std::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
                    this->getElementMeshVertexIndices(zeroNdIndex)));

            const Tlhs eval = [DPhiInvT, absDetDPhi](
                                  const std::size_t& i, const std::size_t& j,
                                  const Vector<Vector<Tlhs, Dim>, NumElementDOFs>& grad_b_q_k) {
                return dot((DPhiInvT * grad_b_q_k[j]), (DPhiInvT * grad_b_q_k[i])).apply()
                       * absDetDPhi;
            };

            algo_m.setOperator([this, eval](lhs_type field) {
                return lagrangeSpace_m.evaluateAx(field.getView(), eval);
            });
            algo_m(*(this->lhs_mp), *(this->rhs_mp), this->params_m);

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
        int getIterationCount() { return algo_m.getIterationCount(); }

        /**
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

        ElementType refElement_m;
        QuadratureType quadrature_m;
        LagrangeSpace<Tlhs, 3, 1, QuadratureType> lagrangeSpace_m;
    };

}  // namespace ippl

#endif