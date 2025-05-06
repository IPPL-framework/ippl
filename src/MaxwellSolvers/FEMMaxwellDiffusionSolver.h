// Class FEMMaxwellDifussionSolver
//   Solves the electric diffusion probelm given by curl(curl(E)) + E = f in 
//   the domain and n x E = 0 on the boundary.

#ifndef IPPL_FEM_MAXWELL_DIFUSSION_SOLVER_H
#define IPPL_FEM_MAXWELL_DIFFUSION_SOLVER_H

#include "LinearSolvers/PCG.h"
#include "Maxwell.h"
#include <iomanip>
#include <iostream>
#include <fstream>

namespace ippl {

    template <typename T, unsigned Dim, unsigned numElementDOFs>
    struct EvalFunctor {
        const Vector<T, Dim> DPhiInvT;
        const T absDetDPhi;

        EvalFunctor(Vector<T, Dim> DPhiInvT, T absDetDPhi)
            : DPhiInvT(DPhiInvT)
            , absDetDPhi(absDetDPhi) {}

        KOKKOS_FUNCTION const auto operator()(size_t i, size_t j,
            const ippl::Vector<ippl::Vector<T, Dim>, numElementDOFs>& curl_b_q_k,
            const ippl::Vector<ippl::Vector<T, Dim>, numElementDOFs>& val_b_q_k, bool onBoundary) const {
            
            //std::cout << "curl_val: " << dot(curl_b_q_k[j], curl_b_q_k[i]).apply() << "\n";
            //std::cout << "non curl val: " << dot(DPhiInvT*val_b_q_k[j], DPhiInvT*val_b_q_k[i]).apply() << "\n";
            //std::cout << absDetDPhi << "\n";
            // 
            T curlTerm = dot(curl_b_q_k[j], curl_b_q_k[i]).apply()/absDetDPhi;
            T massTerm = dot(val_b_q_k[j], val_b_q_k[i]).apply();
            return (curlTerm + massTerm)*absDetDPhi;
        }
    };

    /**
     * @brief A solver for the electric diffusion equation given by
     * \f$ \nabla \times \nabla \times E + E = f \text{ in } \Omega\f$ and
     * \f$ n \times E = 0 \text{ on } \partial \Omega\f$ using the Nédélec basis
     * functions.
     *
     * @tparam FieldType The type used to represent a field on a mesh.
     */
    template <typename FieldType>
    class FEMMaxwellDiffusionSolver : public Maxwell<FieldType, FieldType> {
        constexpr static unsigned Dim = FieldType::dim;

        // we call value_type twice as in theory we expect the field to store
        // vector data represented by an ippl::Vector
        using T = typename FieldType::value_type::value_type;

    public:
        using Base = Maxwell<FieldType, FieldType>;
        using MeshType = typename FieldType::Mesh_t;

        // PCG (Preconditioned Conjugate Gradient) is the solver algorithm used
        using PCGSolverAlgorithm_t = CG<FEMVector<T>, FEMVector<T>, FEMVector<T>, FEMVector<T>,
            FEMVector<T>, FEMVector<T>, FEMVector<T>>;

        // FEM Space types
        using ElementType = std::conditional_t<Dim == 2, ippl::QuadrilateralElement<T>, 
                                ippl::HexahedralElement<T>>;

        using QuadratureType = GaussJacobiQuadrature<T, 5, ElementType>;

        using NedelecType = NedelecSpace<T, Dim, 1, ElementType, QuadratureType, FieldType>;

        // default constructor (compatibility with Alpine)
        FEMMaxwellDiffusionSolver() 
            : Base()
            , rhsVector_m(nullptr)
            , refElement_m()
            , quadrature_m(refElement_m, 0.0, 0.0)
            , nedelecSpace_m(*(new MeshType(NDIndex<Dim>(Vector<unsigned, Dim>(0)), Vector<T, Dim>(0),
                                Vector<T, Dim>(0))), refElement_m, quadrature_m)
        {}

        template <typename F>
        FEMMaxwellDiffusionSolver(FieldType& lhs, FieldType& rhs,
            FEMVector<ippl::Vector<T,Dim>> rhsVectorField, const F& functor)
            : Base(lhs, lhs, rhs)
            , rhsVector_m(nullptr)
            , refElement_m()
            , quadrature_m(refElement_m, 0.0, 0.0)
            , nedelecSpace_m(rhs.get_mesh(), refElement_m, quadrature_m, rhs.getLayout()) {
            
            static_assert(std::is_floating_point<T>::value, "Not a floating point type");
            setDefaultParameters();

            // start a timer
            static IpplTimings::TimerRef init = IpplTimings::getTimer("initFEM");
            IpplTimings::startTimer(init);
            rhsVector_m =
                //std::make_unique<FEMVector<T>>(nedelecSpace_m.evaluateLoadVector(rhsVectorField));
                std::make_unique<FEMVector<T>>(nedelecSpace_m.evaluateLoadVectorFunctor(rhsVectorField, functor));
            /*
            rhs.fillHalo();
            
            // interpolate to the FEMVector
            rhsVector_m = std::make_unique<FEMVector<T>>(
                nedelecSpace_m.interpolateToFEMVector(rhs));
            
            // evaluate the rhs, this will fill the FEMVector
            nedelecSpace_m.evaluateLoadVector(*rhsVector_m);
            
            // do some halo stuff
            rhsVector_m->accumulateHalo();
            rhsVector_m->fillHalo();
            
            // reconstruct to the ippl field, such that this is already
            // accessible before calling solve
            nedelecSpace_m.reconstructToField(*rhsVector_m, rhs);
            */
            IpplTimings::stopTimer(init);
        }

        void setRhs(FieldType& rhs) {
            /*
            Base::setRhs(rhs);

            nedelecSpace_m.initialize(rhs.get_mesh(), rhs.getLayout());

            rhs.fillHalo();
            
            // interpolate to the FEMVector
            rhsVector_m = std::make_unique<FEMVector<T>>(
                nedelecSpace_m.interpolateToFEMVector(rhs));
            
            // evaluate the rhs, this will fill the FEMVector
            nedelecSpace_m.evaluateLoadVector(*rhsVector_m);
            
            // do some halo stuff
            rhsVector_m->accumulateHalo();
            rhsVector_m->fillHalo();
            
            // reconstruct to the ippl field, such that this is already
            // accessible before calling solve
            nedelecSpace_m.reconstructToField(*rhsVector_m, rhs);
            */
        }

        /**
         * @brief Solve the poisson equation using finite element methods.
         * The problem is described by -laplace(lhs) = rhs
         */
        FEMVector<Vector<T,Dim> > solve() {
            // start a timer
            static IpplTimings::TimerRef solve = IpplTimings::getTimer("solve");
            IpplTimings::startTimer(solve);

            const Vector<size_t, Dim> zeroNdIndex = Vector<size_t, Dim>(0);

            // We can pass the zeroNdIndex here, since the transformation
            // jacobian does not depend on translation
            const auto firstElementVertexPoints =
                nedelecSpace_m.getElementMeshVertexPoints(zeroNdIndex);

            // Compute Inverse Transpose Transformation Jacobian ()
            const Vector<T, Dim> DPhiInvT =
                //refElement_m.getTransformationJacobian(firstElementVertexPoints);
                refElement_m.getInverseTransposeTransformationJacobian(firstElementVertexPoints);

            // Compute absolute value of the determinant of the transformation
            // jacobian (|det D Phi_K|)
            const T absDetDPhi = Kokkos::abs(
                refElement_m.getDeterminantOfTransformationJacobian(firstElementVertexPoints));

            EvalFunctor<T, Dim, this->nedelecSpace_m.numElementDOFs> maxwellDiffusionEval(
                DPhiInvT, absDetDPhi);

            const auto algoOperator = [maxwellDiffusionEval, this](FEMVector<T> vector)
                                                                            -> FEMVector<T> {
                // start a timer
                static IpplTimings::TimerRef opTimer = IpplTimings::getTimer("operator");
                IpplTimings::startTimer(opTimer);

                //vector.fillHalo();

                FEMVector<T> return_vector = nedelecSpace_m.evaluateAx(vector,maxwellDiffusionEval);

                //return_vector.accumulateHalo();
                
                IpplTimings::stopTimer(opTimer);

                return return_vector;
            };

            pcg_algo_m.setOperator(algoOperator);

            
                


            // start a timer
            static IpplTimings::TimerRef pcgTimer = IpplTimings::getTimer("pcg");
            IpplTimings::startTimer(pcgTimer);

            //FEMVector<T> lhsVector = lagrangeSpace_m.interpolateToFEMVector(*(this->lhs_mp));
            FEMVector<T> lhsVector = rhsVector_m->deepCopy();
            
            try {
                pcg_algo_m(lhsVector, *rhsVector_m, this->params_m);
            } catch (IpplException& e) {
                std::cout << e.where() << ": " << e.what() << "\n";
                std::cout << "EXITING\n";
                exit(-1);
            }
            
            //lhsVector.fillHalo();

            IpplTimings::stopTimer(pcgTimer);

            
            //lagrangeSpace_m.reconstructToField(lhsVector, *(this->lhs_mp));
            //lagrangeSpace_m.reconstructToField(*rhsVector_m, *(this->rhs_mp));

            IpplTimings::stopTimer(solve);
            lhsVector_m = std::make_unique<FEMVector<T>>(lhsVector);
            return nedelecSpace_m.reconstructBasis(lhsVector);
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
        T getResidue() const { return pcg_algo_m.getResidue(); }

        /**
         * Query the L2-norm error compared to a given (analytical) sol
         * @return L2 error after last solve
         */
        template <typename F>
        T getL2Error(const FEMVector<Vector<T,Dim>>& u, const F& analytic) {
            T error_norm = this->nedelecSpace_m.computeError(u, analytic);
            return error_norm;
        }

        template <typename F>
        T getL2ErrorCoeff(const FEMVector<T>& u, const F& analytic) {
            T error_norm = this->nedelecSpace_m.computeErrorCoeff(u, analytic);
            return error_norm;
        }


        std::unique_ptr<FEMVector<T>> rhsVector_m;
        std::unique_ptr<FEMVector<T>> lhsVector_m;
    protected:
        PCGSolverAlgorithm_t pcg_algo_m;
        

        virtual void setDefaultParameters() {
            this->params_m.add("max_iterations", 10);
            this->params_m.add("tolerance", (T)1e-13);
        }

        ElementType refElement_m;
        QuadratureType quadrature_m;
        NedelecType nedelecSpace_m;
    };

}  // namespace ippl



#endif // IPPL_FEM_MAXWELL_DIFFUSION_SOLVER_H