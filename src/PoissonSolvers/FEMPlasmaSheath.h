// Class FEMPlasmaSheath

#ifndef IPPL_FEMPLASMASHEATH_H
#define IPPL_FEMPLASMASHEATH_H

#include "LinearSolvers/PCG.h"
#include "Poisson.h"

namespace ippl {

    template <typename Tlhs, unsigned Dim, unsigned numElemDOFs>
    struct EvalFunctor {
        const Vector<Tlhs, Dim> DPhiInvT;
        const Tlhs absDetDPhi;
        const Tlhs e_Te;   // e / T_e
        const Tlhs n_inf;  // n_infty
        const Tlhs n_inf;  // n_infty
        FieldLHS& phi_prev; // (phi_(it -1))
        LagrangeType* lagrangeSpace_ptr; // pointer to lagrangeSpace

        EvalFunctor(Vector<Tlhs, Dim> DPhiInvT, Tlhs absDetDPhi, Tlhs e_Te, Tlhs n_inf)
        EvalFunctor(Vector<Tlhs, Dim> DPhiInvT, Tlhs absDetDPhi, Tlhs e_Te, Tlhs n_inf,
                    Tlhs phi_inf, FieldLHS& phi_prev, LagrangeType* lagrangeSpace)
            : DPhiInvT(DPhiInvT)
            , absDetDPhi(absDetDPhi)
            , e_Te(e_Te), n_inf(n_inf)
            , phi_inf(phi_inf), phi_prev(phi_prev)
            , lagrangeSpace_ptr(lagrangeSpace)
            {}

        // TODO: figure out how to pass w^B and interpolate to q_k
        KOKKOS_FUNCTION const auto getInterpolated_wB(unsigned int elemIdx,
            const Vector<Tlhs, numElemDOFs>& b_q_k) {
            const Vector<size_t, numElementDOFs> global_dofs =
                    lagrangeSpace_ptr->getGlobalDOFIndices(elementIndex);

            auto ldom = (phi_prev.getLayout()).getLocalNDIndex();
            const int nghost = phi_prev.getNghost();

            T val = 0;
            for (size_t i = 0; i < numElemDOFs; ++i) {
                auto dof_ndindex = lagrangeSpace_ptr->getMeshVertexNDIndex(gobal_dofs[i]);
                for (unsigned d = 0; d < Dim; ++d) {
                    dof_ndindex[d] = dof_ndindex[d] + nghost - ldom[d].first();
                }
                val += b_q_k[i] * Kokkos::exp(e_Te * (apply(phi_prev, dof_ndindex) - phi_inf));
            }
            return val;
        }


        KOKKOS_FUNCTION const auto operator()(
            const size_t& i, const size_t& j, const Vector<Tlhs, numElemDOFs>& b_q_k,
            const Vector<Vector<Tlhs, Dim>, numElemDOFs>& grad_b_q_k, int elemId, int elemIdxx) const {

                // get wB value (with interpolated phi_prev)
                Tlhs val_w_k = getInterpolated_wB(elemIdx, b_q_k);

                // stiffness matrix term
                Tlhs val1 = dot((DPhiInvT * grad_b_q_k[j]), (DPhiInvT * grad_b_q_k[i])).apply();

                // mass matrix term - needs interpolated w^B(q_k) value
                Tlhs val2 = n_inf * e_Te * b_q_k[i] * b_q_k[j] * val_w_k;

                return (val1 + val2) * absDetDPhi;
        }
    };

    template <typename Tlhs, unsigned Dim, unsigned numElemDOFs>
    struct RHSFunctor {
        const Tlhs absDetDPhi;
        const Tlhs e_Te;   // e / T_e
        const Tlhs n_inf;  // n_infty
        const Tlhs phi_inf;  // phi_infty
        FieldRHS& rho;      // (q_i * n_i - e * n_e)
        FieldLHS& phi_prev; // (phi_(it -1))
        LagrangeType* lagrangeSpace_ptr; // pointer to lagrangeSpace

        EvalFunctor(Tlhs absDetDPhi, Tlhs e_Te, Tlhs n_inf, Tlhs phi_inf, 
                    FieldRHS& rho, FieldLHS& phi_prev, LagrangeType* lagrangeSpace)
            : absDetDPhi(absDetDPhi), e_Te(e_Te)
            , n_inf(n_inf), phi_inf(phi_inf), rho(rho)
            , phi_prev(phi_prev), lagrangeSpace_ptr(lagrangeSpace)
            {}

        // TODO: figure out how to pass w^B and interpolate to q_k
        
        KOKKOS_FUNCTION const auto getLocal(unsigned int elemIdx) {
            Vector<Tlhs, numElemDOFs> rho_local;
            Vector<Tlhs, numElemDOFs> phi_prev_local;

            const Vector<size_t, numElementDOFs> global_dofs =
                    lagrangeSpace_ptr->getGlobalDOFIndices(elementIndex);

            auto ldom = (rho.getLayout()).getLocalNDIndex();
            const int nghost = rho.getNghost();

            for (size_t i = 0; i < numElemDOFs; ++i) {
                auto dof_ndindex = lagrangeSpace_ptr->getMeshVertexNDIndex(gobal_dofs[i]);
                for (unsigned d = 0; d < Dim; ++d) {
                    dof_ndindex[d] = dof_ndindex[d] + nghost - ldom[d].first();
                }
                rho_local[i] = apply(rho_local, dof_ndindex);
                phi_prev_local[i] = apply(phi_prev, dof_ndindex);
            }

            return {rho_local, phi_prev_local};
        }
        
        KOKKOS_FUNCTION const auto getInterpolated_wB(unsigned int elemIdx,
            const Vector<Tlhs, numElemDOFs>& b_q_k) {
            const Vector<size_t, numElementDOFs> global_dofs =
                    lagrangeSpace_ptr->getGlobalDOFIndices(elementIndex);

            auto ldom = (phi_prev.getLayout()).getLocalNDIndex();
            const int nghost = phi_prev.getNghost();

            T val = 0;
            for (size_t i = 0; i < numElemDOFs; ++i) {
                auto dof_ndindex = lagrangeSpace_ptr->getMeshVertexNDIndex(gobal_dofs[i]);
                for (unsigned d = 0; d < Dim; ++d) {
                    dof_ndindex[d] = dof_ndindex[d] + nghost - ldom[d].first();
                }
                val += b_q_k[i] * Kokkos::exp(e_Te * (apply(phi_prev, dof_ndindex) - phi_inf));
            }
            return val;
        }

        KOKKOS_FUNCTION const auto operator()(
            const size_t& i, const Vector<Tlhs, numElemDOFs>& b_q_k, int elemIdx) const {

                Vector<Tlhs, numElemDOFs> rho_local = getLocal(elemIdx).first;
                Vector<Tlhs, numElemDOFs> phi_prev_local = getLocal(elemIdx).second;
                Tlhs val_w_k = getInterpolated_wB(elemIdx, b_q_k);

                Tlhs val;
                for (size_t j = 0; j < numElemDOFs; ++j) {
                    val += b_q_k[i] * b_q_k[j] * rho_local[j]
                           + n_inf * e_Te * phi_prev_local[j] * val_w_k;
                }
                return val * absDetDPhi;
        }
    };

    /**
     * @brief A solver for the poisson equation using finite element methods and
     * Conjugate Gradient (CG)
     *
     * @tparam FieldLHS field type for the left hand side
     * @tparam FieldRHS field type for the right hand side
     */
    template <typename FieldLHS, typename FieldRHS = FieldLHS, unsigned Order = 1, unsigned QuadNumNodes = 5>
    class FEMPoissonSolver : public Poisson<FieldLHS, FieldRHS> {
        constexpr static unsigned Dim = FieldLHS::dim;
        using Tlhs                    = typename FieldLHS::value_type;

    public:
        using Base = Poisson<FieldLHS, FieldRHS>;
        using typename Base::lhs_type, typename Base::rhs_type;
        using MeshType = typename FieldRHS::Mesh_t;

        // PCG (Preconditioned Conjugate Gradient) is the solver algorithm used
        using PCGSolverAlgorithm_t =
            CG<lhs_type, lhs_type, lhs_type, lhs_type, lhs_type, FieldLHS, FieldRHS>;

        // FEM Space types
        using ElementType =
            std::conditional_t<Dim == 1, ippl::EdgeElement<Tlhs>,
                               std::conditional_t<Dim == 2, ippl::QuadrilateralElement<Tlhs>,
                                                  ippl::HexahedralElement<Tlhs>>>;

        using QuadratureType = GaussJacobiQuadrature<Tlhs, QuadNumNodes, ElementType>;

        using LagrangeType = LagrangeSpace<Tlhs, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>;

        // default constructor (compatibility with Alpine)
        FEMPoissonSolver() 
            : Base()
            , refElement_m()
            , quadrature_m(refElement_m, 0.0, 0.0)
            , lagrangeSpace_m(*(new MeshType(NDIndex<Dim>(Vector<unsigned, Dim>(0)), Vector<Tlhs, Dim>(0),
                                Vector<Tlhs, Dim>(0))), refElement_m, quadrature_m)
        {}

        FEMPoissonSolver(lhs_type& lhs, rhs_type& rhs)
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

            EvalFunctor<Tlhs, Dim, this->lagrangeSpace_m.numElementDOFs> poissonEquationEval(
                DPhiInvT, absDetDPhi);

            // get BC type of our RHS
            BConds<FieldRHS, Dim>& bcField = (this->rhs_mp)->getFieldBC();
            FieldBC bcType = bcField[0]->getBCType();

            const auto algoOperator = [poissonEquationEval, &bcField, this](rhs_type field) -> lhs_type {
                // start a timer
                static IpplTimings::TimerRef opTimer = IpplTimings::getTimer("operator");
                IpplTimings::startTimer(opTimer);

                // set appropriate BCs for the field as the info gets lost in the CG iteration
                field.setFieldBC(bcField);

                field.fillHalo();

                auto return_field = lagrangeSpace_m.evaluateAx(field, poissonEquationEval);

                IpplTimings::stopTimer(opTimer);

                return return_field;
            };

            pcg_algo_m.setOperator(algoOperator);

            // send boundary values to RHS (load vector) i.e. lifting (Dirichlet BCs)
            if (bcType == CONSTANT_FACE) {
                *(this->rhs_mp) = *(this->rhs_mp) -
                    lagrangeSpace_m.evaluateAx_lift(*(this->rhs_mp), poissonEquationEval);
            }

            // start a timer
            static IpplTimings::TimerRef pcgTimer = IpplTimings::getTimer("pcg");
            IpplTimings::startTimer(pcgTimer);

            pcg_algo_m(*(this->lhs_mp), *(this->rhs_mp), this->params_m);

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
