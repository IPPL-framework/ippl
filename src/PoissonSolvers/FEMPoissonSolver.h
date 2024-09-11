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
        using PCGSolverAlgorithm_t = CG<lhs_type, lhs_type, lhs_type, lhs_type,
                                         lhs_type, FieldLHS, FieldRHS>;

        // FEM Space types
        using ElementType =
            std::conditional_t<Dim == 1, ippl::EdgeElement<Tlhs>,
                               std::conditional_t<Dim == 2, ippl::QuadrilateralElement<Tlhs>,
                                                  ippl::HexahedralElement<Tlhs>>>;

        using QuadratureType = GaussJacobiQuadrature<Tlhs, 5, ElementType>;

        FEMPoissonSolver(lhs_type& lhs, rhs_type& rhs)
            : Base(lhs, rhs)
            , refElement_m()
            , quadrature_m(refElement_m, 0.0, 0.0)
            , lagrangeSpace_m(lhs.get_mesh(), refElement_m, quadrature_m, rhs.getLayout()) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();

            // start a timer
            static IpplTimings::TimerRef init = IpplTimings::getTimer("initFEM");
            IpplTimings::startTimer(init);

            Inform m("");

            m << "inside constructor" << endl;

            rhs.fillHalo();

            m << "after fillHalo" << endl;

            //rhs.write();

            lagrangeSpace_m.evaluateLoadVector(rhs);

            m << "after eval load vector" << endl;

            rhs.accumulateHalo();
            rhs.fillHalo();

            m << "after accumulate Halo" << endl;
            //rhs.write();
            
            IpplTimings::stopTimer(init);
        }

        /**
         * @brief Solve the poisson equation using finite element methods.
         * The problem is described by -laplace(lhs) = rhs
         */
        void solve() override {

            // start a timer
            static IpplTimings::TimerRef solve = IpplTimings::getTimer("solve");
            IpplTimings::startTimer(solve);

            const Vector<size_t, Dim> zeroNdIndex = Vector<size_t, Dim>(0);

            // We can pass the zeroNdIndex here, since the transformation jacobian does not depend
            // on translation
            const auto firstElementVertexPoints =
                lagrangeSpace_m.getElementMeshVertexPoints(zeroNdIndex);

            // Compute Inverse Transpose Transformation Jacobian ()
            const Vector<Tlhs, Dim> DPhiInvT =
                refElement_m.getInverseTransposeTransformationJacobian(firstElementVertexPoints);

            // Compute absolute value of the determinant of the transformation jacobian (|det D
            // Phi_K|)
            const Tlhs absDetDPhi = Kokkos::abs(
                refElement_m.getDeterminantOfTransformationJacobian(firstElementVertexPoints));

            const auto poissonEquationEval =
                [this, DPhiInvT, absDetDPhi](
                    const size_t& i, const size_t& j,
                    const Vector<Vector<Tlhs, Dim>, this->lagrangeSpace_m.numElementDOFs>&
                        grad_b_q_k) {
                    return dot((DPhiInvT * grad_b_q_k[j]), (DPhiInvT * grad_b_q_k[i])).apply()
                           * absDetDPhi;
                };

            const auto algoOperator = [poissonEquationEval,
                                       this](lhs_type field) -> lhs_type {

                // start a timer
                static IpplTimings::TimerRef opTimer = IpplTimings::getTimer("operator");
                IpplTimings::startTimer(opTimer);

                field.fillHalo();

                auto return_field = lagrangeSpace_m.evaluateAx(field, poissonEquationEval);
            
                return_field.accumulateHalo();
                //return_field.fillHalo();
            
                IpplTimings::stopTimer(opTimer);

                return return_field;
            };

            pcg_algo_m.setOperator(algoOperator);

            // start a timer
            static IpplTimings::TimerRef pcgTimer = IpplTimings::getTimer("pcg");
            IpplTimings::startTimer(pcgTimer);

            pcg_algo_m(*(this->lhs_mp), *(this->rhs_mp), this->params_m);

            IpplTimings::stopTimer(pcgTimer);

            int output = this->params_m.template get<int>("output_type");
            if (output & Base::GRAD) {
                *(this->grad_mp) = -grad(*(this->lhs_mp));
            }

            IpplTimings::stopTimer(solve);
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
        LagrangeSpace<Tlhs, Dim, 1, ElementType, QuadratureType, FieldLHS, FieldRHS> lagrangeSpace_m;
    };

}  // namespace ippl

#endif
