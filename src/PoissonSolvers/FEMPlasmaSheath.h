// Class FEMPlasmaSheath

#ifndef IPPL_FEMPLASMASHEATH_H
#define IPPL_FEMPLASMASHEATH_H

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
    template <typename FieldLHS, typename FieldRHS = FieldLHS, unsigned Order = 1, unsigned QuadNumNodes = 5>
    class FEMPlasmaSheath : public Poisson<FieldLHS, FieldRHS> {
        constexpr static unsigned Dim = FieldLHS::dim;
        using Tlhs                    = typename FieldLHS::value_type;

    public:
        using Base = Poisson<FieldLHS, FieldRHS>;
        using typename Base::lhs_type, typename Base::rhs_type;
        using MeshType = typename FieldRHS::Mesh_t;
        using ViewType = typename detail::ViewType<Tlhs, Dim>::view_type;

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

        // structs used for matrix-free evaluation of A and b (matrix, load vector)
        
        template <unsigned numElemDOFs>
        struct EvalFunctor {
            const Vector<Tlhs, Dim> DPhiInvT;
            const Tlhs absDetDPhi;
            const Tlhs e_Te;   // e / T_e
            const Tlhs n_inf;  // n_infty
            const Tlhs phi_inf; // phi_infty
            ViewType phi_prev; // (phi_(it -1))
            LagrangeType lagrangeSpace; // lagrangeSpace

            EvalFunctor(Vector<Tlhs, Dim> DPhiInvT, Tlhs absDetDPhi, Tlhs e_Te, Tlhs n_inf,
                        Tlhs phi_inf, ViewType& phi_prev, LagrangeType lagrangeSpace)
                : DPhiInvT(DPhiInvT)
                , absDetDPhi(absDetDPhi)
                , e_Te(e_Te), n_inf(n_inf)
                , phi_inf(phi_inf), phi_prev(phi_prev)
                , lagrangeSpace(lagrangeSpace)
                {}

            KOKKOS_FUNCTION const auto operator()(
                const size_t& i, const size_t& j, const Vector<Tlhs, numElemDOFs>& b_q_k,
                const Vector<Vector<Tlhs, Dim>, numElemDOFs>& grad_b_q_k, int elemIdx) const {

                    const Vector<size_t, numElemDOFs> global_dofs =
                            lagrangeSpace.getGlobalDOFIndices(elemIdx);

                    Tlhs val_w_k = 0;
                    for (size_t s = 0; s < numElemDOFs; ++s) {
                        auto dof_ndindex = lagrangeSpace.getMeshVertexNDIndex(global_dofs[s]);
                        for (unsigned d = 0; d < Dim; ++d) {
                            dof_ndindex[d] = dof_ndindex[d] + 1; //+ nghost - ldom[d].first();
                        }
                        val_w_k += b_q_k[s] * Kokkos::exp(e_Te * (apply(phi_prev, dof_ndindex) - phi_inf));
                    }

                    // stiffness matrix term
                    Tlhs val1 = dot((DPhiInvT * grad_b_q_k[j]), (DPhiInvT * grad_b_q_k[i])).apply();

                    // mass matrix term - needs interpolated w^B(q_k) value
                    Tlhs val2 = n_inf * e_Te * b_q_k[i] * b_q_k[j] * val_w_k;

                    return (val1 + val2) * absDetDPhi;
            }
        };

        template <unsigned numElemDOFs>
        struct RHSFunctor {
            const Tlhs absDetDPhi;
            const Tlhs e_Te;   // e / T_e
            const Tlhs n_inf;  // n_infty
            const Tlhs phi_inf;  // phi_infty
            ViewType rho;      // (q_i * n_i - e * n_e)
            ViewType phi_prev; // (phi_(it -1))
            LagrangeType lagrangeSpace; // lagrangeSpace

            RHSFunctor(Tlhs absDetDPhi, Tlhs e_Te, Tlhs n_inf, Tlhs phi_inf, 
                        ViewType& rho, ViewType& phi_prev, LagrangeType lagrangeSpace)
                : absDetDPhi(absDetDPhi), e_Te(e_Te)
                , n_inf(n_inf), phi_inf(phi_inf), rho(rho)
                , phi_prev(phi_prev), lagrangeSpace(lagrangeSpace)
                {}

            KOKKOS_FUNCTION const auto operator()(
                const size_t& i, const Vector<Tlhs, numElemDOFs>& b_q_k, unsigned int elemIdx) const {

                    const Vector<size_t, numElemDOFs> global_dofs =
                            lagrangeSpace.getGlobalDOFIndices(elemIdx);

                    Tlhs val_w_k = 0;
                    for (size_t j = 0; j < numElemDOFs; ++j) {
                        auto dof_ndindex = lagrangeSpace.getMeshVertexNDIndex(global_dofs[j]);
                        for (unsigned d = 0; d < Dim; ++d) {
                            dof_ndindex[d] = dof_ndindex[d] + 1; //+ nghost - ldom[d].first();
                        }
                        val_w_k += b_q_k[j] * Kokkos::exp(e_Te * (apply(phi_prev, dof_ndindex) - phi_inf));
                    }

                    Tlhs val = 0;
                    for (size_t j = 0; j < numElemDOFs; ++j) {
                        auto dof_ndindex = lagrangeSpace.getMeshVertexNDIndex(global_dofs[j]);
                        for (unsigned d = 0; d < Dim; ++d) {
                            dof_ndindex[d] = dof_ndindex[d] + 1;// nghost - ldom[d].first();
                        }
                        Tlhs rho_loc = apply(rho, dof_ndindex);
                        Tlhs phi_prev_loc = apply(phi_prev, dof_ndindex);

                        val += b_q_k[i] * b_q_k[j] * (rho_loc
                               + n_inf * e_Te * phi_prev_loc * val_w_k);
                    }
                    return val * absDetDPhi;
            }
        };

        // FEMPlasmaSheath
        // default constructor (compatibility with Alpine)
        FEMPlasmaSheath() 
            : Base()
            , refElement_m()
            , quadrature_m(refElement_m, 0.0, 0.0)
            , lagrangeSpace_m(*(new MeshType(NDIndex<Dim>(Vector<unsigned, Dim>(0)), Vector<Tlhs, Dim>(0),
                                Vector<Tlhs, Dim>(0))), refElement_m, quadrature_m)
            , e_Te(0.0), n_inf(0.0), phi_inf(0.0)
        {}

        FEMPlasmaSheath(lhs_type& lhs, rhs_type& rhs, Tlhs e_Te, Tlhs n_inf, Tlhs phi_inf)
            : Base(lhs, rhs)
            , refElement_m()
            , quadrature_m(refElement_m, 0.0, 0.0)
            , lagrangeSpace_m(rhs.get_mesh(), refElement_m, quadrature_m, rhs.getLayout())
            , e_Te(e_Te), n_inf(n_inf), phi_inf(phi_inf)
        {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");

            Inform m("");

            // need to pass rho = q_i*n_i - q_e*n_e as rhs
            // and phi_prev as lhs

            setDefaultParameters();

            // start a timer
            static IpplTimings::TimerRef init = IpplTimings::getTimer("initFEM");
            IpplTimings::startTimer(init);
            
            const Vector<size_t, Dim> zeroNdIndex = Vector<size_t, Dim>(0);

            // We can pass the zeroNdIndex here, since the transformation jacobian does not depend
            // on translation
            const auto firstElementVertexPoints =
                lagrangeSpace_m.getElementMeshVertexPoints(zeroNdIndex);

            // Compute absolute value of the determinant of the transformation jacobian (|det D
            // Phi_K|)
            const Tlhs absDetDPhi = Kokkos::abs(
                refElement_m.getDeterminantOfTransformationJacobian(firstElementVertexPoints));

            // initialize the RHSFunctor struct to pass to Load vector creation
            RHSFunctor<this->lagrangeSpace_m.numElementDOFs> modifiedPoissonRHS(
                absDetDPhi, e_Te, n_inf, phi_inf, (this->rhs_mp)->getView(),
                (this->lhs_mp)->getView(), this->lagrangeSpace_m);

            rhs.fillHalo();

            lagrangeSpace_m.evaluateLoadVector(rhs, modifiedPoissonRHS);

            rhs.fillHalo();

            IpplTimings::stopTimer(init);
        }

        void setRhs(rhs_type& rhs) override {
            Base::setRhs(rhs);

            if (this->lhs_mp == NULL) {
                throw IpplException("FEMPlasmaSheath::setRhs(rhs_type&)", 
                "No Lhs set! Please set the Lhs before calling setRhs");
            }

            lagrangeSpace_m.initialize(rhs.get_mesh(), rhs.getLayout());

            const Vector<size_t, Dim> zeroNdIndex = Vector<size_t, Dim>(0);

            // We can pass the zeroNdIndex here, since the transformation jacobian does not depend
            // on translation
            const auto firstElementVertexPoints =
                lagrangeSpace_m.getElementMeshVertexPoints(zeroNdIndex);

            // Compute absolute value of the determinant of the transformation jacobian (|det D
            // Phi_K|)
            const Tlhs absDetDPhi = Kokkos::abs(
                refElement_m.getDeterminantOfTransformationJacobian(firstElementVertexPoints));

            // initialize the RHSFunctor struct to pass to Load vector creation
            //RHSFunctor<this->lagrangeSpace_m.numElementDOFs> modifiedPoissonRHS(
            //    absDetDPhi, e_Te, n_inf, phi_inf, *this->rhs_mp, *this->lhs_mp, this->lagrangeSpace_m);

            rhs.fillHalo();

            //lagrangeSpace_m.evaluateLoadVector(rhs, modifiedPoissonRHS);

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

            EvalFunctor<this->lagrangeSpace_m.numElementDOFs> modifiedPoissonEval(
                DPhiInvT, absDetDPhi, e_Te, n_inf, phi_inf, (this->lhs_mp)->getView(), this->lagrangeSpace_m);

            // get BC type of our RHS
            BConds<FieldRHS, Dim>& bcField = (this->rhs_mp)->getFieldBC();
            FieldBC bcType = bcField[0]->getBCType();

            const auto algoOperator = [modifiedPoissonEval, &bcField, this](rhs_type field) -> lhs_type {
                // start a timer
                static IpplTimings::TimerRef opTimer = IpplTimings::getTimer("operator");
                IpplTimings::startTimer(opTimer);

                // set appropriate BCs for the field as the info gets lost in the CG iteration
                field.setFieldBC(bcField);

                field.fillHalo();

                auto return_field = lagrangeSpace_m.evaluateAx(field, modifiedPoissonEval);

                IpplTimings::stopTimer(opTimer);

                return return_field;
            };

            pcg_algo_m.setOperator(algoOperator);

            // send boundary values to RHS (load vector) i.e. lifting (Dirichlet BCs)
            if (bcType == CONSTANT_FACE) {
                *(this->rhs_mp) = *(this->rhs_mp) -
                    lagrangeSpace_m.evaluateAx_lift(*(this->rhs_mp), modifiedPoissonEval);
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

        Tlhs e_Te;
        Tlhs n_inf;
        Tlhs phi_inf;
    };

}  // namespace ippl

#endif
