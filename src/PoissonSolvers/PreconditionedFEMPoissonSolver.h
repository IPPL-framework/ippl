// Class FEMPoissonSolver
//   Solves the poisson equation using finite element methods and Conjugate
//   Gradient + a Preconditioner.

#ifndef IPPL_PRECONFEMPOISSONSOLVER_H
#define IPPL_PRECONFEMPOISSONSOLVER_H

// #include "FEM/FiniteElementSpace.h"
#include "LaplaceHelpers.h"
#include "LinearSolvers/PCG.h"
#include "Poisson.h"

namespace ippl {

    template <typename Tlhs, unsigned Dim, unsigned numElemDOFs>
    struct EvalFunctor {
        const Vector<Tlhs, Dim> DPhiInvT;
        const Tlhs absDetDPhi;

        EvalFunctor(Vector<Tlhs, Dim> DPhiInvT, Tlhs absDetDPhi)
            : DPhiInvT(DPhiInvT)
            , absDetDPhi(absDetDPhi) {}

        KOKKOS_FUNCTION auto operator()(
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
    class PreconditionedFEMPoissonSolver : public Poisson<FieldLHS, FieldRHS> {
        constexpr static unsigned Dim = FieldLHS::dim;
        using Tlhs                    = typename FieldLHS::value_type;

    public:
        using Base = Poisson<FieldLHS, FieldRHS>;
        using typename Base::lhs_type, typename Base::rhs_type;
        using MeshType = typename FieldRHS::Mesh_t;

        // PCG (Preconditioned Conjugate Gradient) is the solver algorithm used
        using PCGSolverAlgorithm_t =
            PCG<lhs_type, lhs_type, lhs_type, lhs_type, lhs_type, FieldLHS, FieldRHS>;

        // FEM Space types
        using ElementType =
            std::conditional_t<Dim == 1, ippl::EdgeElement<Tlhs>,
                               std::conditional_t<Dim == 2, ippl::QuadrilateralElement<Tlhs>,
                                                  ippl::HexahedralElement<Tlhs>>>;

        using QuadratureType = GaussJacobiQuadrature<Tlhs, 5, ElementType>;

        using LagrangeType = LagrangeSpace<Tlhs, Dim, 1, ElementType, QuadratureType, FieldLHS, FieldRHS>;

        // default constructor (compatibility with Alpine)
        PreconditionedFEMPoissonSolver() 
            : Base()
            , refElement_m()
            , quadrature_m(refElement_m, 0.0, 0.0)
            , lagrangeSpace_m(*(new MeshType(NDIndex<Dim>(Vector<unsigned, Dim>(0)), Vector<Tlhs, Dim>(0),
                                Vector<Tlhs, Dim>(0))), refElement_m, quadrature_m)
        {}

        PreconditionedFEMPoissonSolver(lhs_type& lhs, rhs_type& rhs)
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

            lagrangeSpace_m.evaluateLoadVector(rhs);

            rhs.fillHalo();
            
            IpplTimings::stopTimer(init);
        }

        void setRhs(rhs_type& rhs) override {
            Base::setRhs(rhs);

            lagrangeSpace_m.initialize(rhs.get_mesh(), rhs.getLayout());

            rhs.fillHalo();

            lagrangeSpace_m.evaluateLoadVector(rhs);

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

            EvalFunctor<Tlhs, Dim, LagrangeType::numElementDOFs> poissonEquationEval(
                DPhiInvT, absDetDPhi);

            // get BC type of our RHS
            BConds<FieldRHS, Dim>& bcField = (this->rhs_mp)->getFieldBC();
            FieldBC bcType = bcField[0]->getBCType();

            const auto algoOperator = [poissonEquationEval, &bcField, this](rhs_type field) -> lhs_type {
                // set appropriate BCs for the field as the info gets lost in the CG iteration
                field.setFieldBC(bcField);

                field.fillHalo();

                auto return_field = lagrangeSpace_m.evaluateAx(field, poissonEquationEval);

                return return_field;
            };

            const auto algoOperatorL = [poissonEquationEval, &bcField, this](lhs_type field) -> lhs_type {
                // set appropriate BCs for the field as the info gets lost in the CG iteration
                field.setFieldBC(bcField);

                field.fillHalo();

                auto return_field = lagrangeSpace_m.evaluateAx_lower(field, poissonEquationEval);

                return return_field;
            };

            const auto algoOperatorU = [poissonEquationEval, &bcField, this](lhs_type field) -> lhs_type {
                // set appropriate BCs for the field as the info gets lost in the CG iteration
                field.setFieldBC(bcField);

                field.fillHalo();

                auto return_field = lagrangeSpace_m.evaluateAx_upper(field, poissonEquationEval);

                return return_field;
            };

            const auto algoOperatorUL = [poissonEquationEval, &bcField, this](lhs_type field) -> lhs_type {
                // set appropriate BCs for the field as the info gets lost in the CG iteration
                field.setFieldBC(bcField);

                field.fillHalo();

                auto return_field = lagrangeSpace_m.evaluateAx_upperlower(field, poissonEquationEval);

                return return_field;
            };

            const auto algoOperatorInvD = [poissonEquationEval, &bcField, this](lhs_type field) -> lhs_type {
                // set appropriate BCs for the field as the info gets lost in the CG iteration
                field.setFieldBC(bcField);

                field.fillHalo();

                auto return_field = lagrangeSpace_m.evaluateAx_inversediag(field, poissonEquationEval);

                return return_field;
            };

            const auto algoOperatorD = [poissonEquationEval, &bcField, this](lhs_type field) -> lhs_type {
                // set appropriate BCs for the field as the info gets lost in the CG iteration
                field.setFieldBC(bcField);

                field.fillHalo();

                auto return_field = lagrangeSpace_m.evaluateAx_diag(field, poissonEquationEval);

                return return_field;
            };

            // set preconditioner for PCG
            std::string preconditioner_type =
                this->params_m.template get<std::string>("preconditioner_type");
            int level    = this->params_m.template get<int>("newton_level");
            int degree   = this->params_m.template get<int>("chebyshev_degree");
            int inner    = this->params_m.template get<int>("gauss_seidel_inner_iterations");
            int outer    = this->params_m.template get<int>("gauss_seidel_outer_iterations");
            double omega = this->params_m.template get<double>("ssor_omega");
            int richardson_iterations =
                this->params_m.template get<int>("richardson_iterations");

            pcg_algo_m.setPreconditioner(algoOperator, algoOperatorL, algoOperatorU, algoOperatorUL,
                                     algoOperatorInvD, algoOperatorD, 0, 0, preconditioner_type,
                                     level, degree, richardson_iterations, inner, outer, omega);

            pcg_algo_m.setOperator(algoOperator);

            // send boundary values to RHS (load vector) i.e. lifting (Dirichlet BCs)
            if (bcType == CONSTANT_FACE) {
                *(this->rhs_mp) = *(this->rhs_mp) -
                    lagrangeSpace_m.evaluateAx_lift(*(this->rhs_mp), poissonEquationEval);
            }

            // start a timer
            static IpplTimings::TimerRef pcgTimer = IpplTimings::getTimer("pcg");
            IpplTimings::startTimer(pcgTimer);

            // run PCG -> lhs contains solution
            pcg_algo_m(*(this->lhs_mp), *(this->rhs_mp), this->params_m);

            // added for BCs to be imposed properly
            // (they are not propagated through the preconditioner)
            if (bcType == CONSTANT_FACE) {
                bcField.assignGhostToPhysical(*(this->lhs_mp));
            }
            (this->lhs_mp)->fillHalo();

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

        /**
         * Query the L2-norm error compared to a given (analytical) sol
         * @return L2 error after last solve
         */
        template <typename F>
        Tlhs getL2Error(const F& analytic) {
            Tlhs error_norm = this->lagrangeSpace_m.computeErrorL2(*(this->lhs_mp), analytic);
            return error_norm;
        }

        /**
         * Query the average of the solution
         * @param vol Boolean indicating whether we divide by volume or not
         * @return avg (offset for null space test cases if divided by volume)
         */
        Tlhs getAvg(bool Vol = false) {
            Tlhs avg = this->lagrangeSpace_m.computeAvg(*(this->lhs_mp));
            if (Vol) {
                lhs_type unit((this->lhs_mp)->get_mesh(), (this->lhs_mp)->getLayout());
                unit = 1.0;
                Tlhs vol = this->lagrangeSpace_m.computeAvg(unit);
                return avg/vol;
            } else {
                return avg;
            }
        }

    protected:
        PCGSolverAlgorithm_t pcg_algo_m;

        virtual void setDefaultParameters() override {
            this->params_m.add("max_iterations", 1000);
            this->params_m.add("tolerance", (Tlhs)1e-13);
        }

        ElementType refElement_m;
        QuadratureType quadrature_m;
        LagrangeType lagrangeSpace_m;
    };

}  // namespace ippl

#endif
