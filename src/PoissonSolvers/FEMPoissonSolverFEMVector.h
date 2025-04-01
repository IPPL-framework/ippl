// Class FEMPoissonSolverFEMVector
//   Solves the poisson equation using finite element methods and Conjugate
//   Gradient

#ifndef IPPL_FEMPOISSONSOLVER_FEMVector_H
#define IPPL_FEMPOISSONSOLVER_FEMVector_H

// #include "FEM/FiniteElementSpace.h"
#include "LinearSolvers/PCG.h"
#include "Poisson.h"
#include <iomanip>
#include <iostream>

namespace ippl {

    template <typename Tlhs, unsigned Dim, unsigned numElemDOFs>
    struct EvalFunctor {
        const Vector<Tlhs, Dim> DPhiInvT;
        const Tlhs absDetDPhi;

        EvalFunctor(Vector<Tlhs, Dim> DPhiInvT, Tlhs absDetDPhi)
            : DPhiInvT(DPhiInvT)
            , absDetDPhi(absDetDPhi) {}

        KOKKOS_FUNCTION const auto operator()(
            const size_t& i, const size_t& j,
            const Vector<Vector<Tlhs, Dim>, numElemDOFs>& grad_b_q_k) const {
            return dot((DPhiInvT * grad_b_q_k[j]), (DPhiInvT * grad_b_q_k[i])).apply() * absDetDPhi;
        }
    };

    /**
     * @brief A solver for the poisson equation using finite element methods and
     * Conjugate Gradient (CG)
     *
     * @tparam FieldLHS field type for the left hand side
     * @tparam FieldRHS field type for the right hand side
     */
    template <typename FieldLHS, typename FieldRHS = FieldLHS>
    class FEMPoissonSolverFEMVector : public Poisson<FieldLHS, FieldRHS> {
        constexpr static unsigned Dim = FieldLHS::dim;
        using Tlhs                    = typename FieldLHS::value_type;

    public:
        using Base = Poisson<FieldLHS, FieldRHS>;
        using typename Base::lhs_type, typename Base::rhs_type;
        using MeshType = typename FieldRHS::Mesh_t;

        // PCG (Preconditioned Conjugate Gradient) is the solver algorithm used
        using PCGSolverAlgorithm_t =
            CG<FEMVector<Tlhs>, FEMVector<Tlhs>, FEMVector<Tlhs>, FEMVector<Tlhs>, FEMVector<Tlhs>, FEMVector<Tlhs>, FEMVector<Tlhs>>;

        // FEM Space types
        using ElementType =
            std::conditional_t<Dim == 1, ippl::EdgeElement<Tlhs>,
                               std::conditional_t<Dim == 2, ippl::QuadrilateralElement<Tlhs>,
                                                  ippl::HexahedralElement<Tlhs>>>;

        using QuadratureType = GaussJacobiQuadrature<Tlhs, 5, ElementType>;

        using LagrangeType = LagrangeSpaceFEMVector<Tlhs, Dim, 1, ElementType, QuadratureType, FEMVector<Tlhs>, FieldRHS>;

        // default constructor (compatibility with Alpine)
        FEMPoissonSolverFEMVector() 
            : Base()
            , refElement_m()
            , quadrature_m(refElement_m, 0.0, 0.0)
            , lagrangeSpace_m(*(new MeshType(NDIndex<Dim>(Vector<unsigned, Dim>(0)), Vector<Tlhs, Dim>(0),
                                Vector<Tlhs, Dim>(0))), refElement_m, quadrature_m)
        {}

        FEMPoissonSolverFEMVector(lhs_type& lhs, rhs_type& rhs)
            : Base(lhs, rhs)
            , refElement_m()
            , quadrature_m(refElement_m, 0.0, 0.0)
            , lagrangeSpace_m(rhs.get_mesh(), refElement_m, quadrature_m, rhs.getLayout()) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();

            // start a timer
            static IpplTimings::TimerRef init = IpplTimings::getTimer("initFEM");
            IpplTimings::startTimer(init);
            
            rhs.fillHalo();
            
            FEMVector<Tlhs> rhsVector = lagrangeSpace_m.interpolateToFEMVector(rhs, rhs.getLayout());
            lagrangeSpace_m.evaluateLoadVector(rhsVector);
            lagrangeSpace_m.reconstructToField(rhsVector, rhs);

            rhs.accumulateHalo();
            rhs.fillHalo();
            
            IpplTimings::stopTimer(init);
        }

        void setRhs(rhs_type& rhs) override {
            Base::setRhs(rhs);

            lagrangeSpace_m.initialize(rhs.get_mesh(), rhs.getLayout());

            rhs.fillHalo();

            FEMVector<Tlhs> rhsVector = lagrangeSpace_m.interpolateToFEMVector(rhs, rhs.getLayout());
            lagrangeSpace_m.evaluateLoadVector(rhsVector);
            lagrangeSpace_m.reconstructToField(rhsVector, rhs);

            rhs.accumulateHalo();
            rhs.fillHalo();
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

            EvalFunctor<Tlhs, Dim, this->lagrangeSpace_m.numElementDOFs> poissonEquationEval(
                DPhiInvT, absDetDPhi);

            const auto algoOperator = [poissonEquationEval, this](FEMVector<Tlhs> vector)
                                                                            -> FEMVector<Tlhs> {
                // start a timer
                static IpplTimings::TimerRef opTimer = IpplTimings::getTimer("operator");
                IpplTimings::startTimer(opTimer);

                vector.fillHalo();

                FEMVector<Tlhs> return_vector = lagrangeSpace_m.evaluateAx(vector,
                    poissonEquationEval);

                return_vector.accumulateHalo();
                
                IpplTimings::stopTimer(opTimer);

                return return_vector;
            };

            pcg_algo_m.setOperator(algoOperator);

            
                


            // start a timer
            static IpplTimings::TimerRef pcgTimer = IpplTimings::getTimer("pcg");
            IpplTimings::startTimer(pcgTimer);

            FEMVector<Tlhs> lhsVector = lagrangeSpace_m.interpolateToFEMVector(*(this->lhs_mp),
                                                            this->lhs_mp->getLayout());
            FEMVector<Tlhs> rhsVector = lagrangeSpace_m.interpolateToFEMVector(*(this->rhs_mp),
                                                            this->rhs_mp->getLayout());
            
            try {
                pcg_algo_m(lhsVector, rhsVector, this->params_m);
            } catch (IpplException& e) {
                std::cout << e.what() << "\n";
            }
            
            lhsVector.fillHalo();

            IpplTimings::stopTimer(pcgTimer);

            
            lagrangeSpace_m.reconstructToField(lhsVector, *(this->lhs_mp));
            lagrangeSpace_m.reconstructToField(rhsVector, *(this->rhs_mp));


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

        /**
         * Query the L2-norm error compared to a given (analytical) sol
         * @return L2 error after last solve
         */
        template <typename F>
        Tlhs getL2Error(const F& analytic) {
            Tlhs error_norm = this->lagrangeSpace_m.computeError(*(this->lhs_mp), analytic);
            return error_norm;
        }

    protected:
        PCGSolverAlgorithm_t pcg_algo_m;

        virtual void setDefaultParameters() override {
            this->params_m.add("max_iterations", 10);
            this->params_m.add("tolerance", (Tlhs)1e-13);
        }

        ElementType refElement_m;
        QuadratureType quadrature_m;
        LagrangeType lagrangeSpace_m;
    };

}  // namespace ippl

#endif
