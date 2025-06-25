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
            
            T curlTerm = dot(DPhiInvT*curl_b_q_k[j], DPhiInvT*curl_b_q_k[i]).apply();
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

        typedef Vector<T,Dim> point_t;

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

        FEMMaxwellDiffusionSolver(FieldType& lhs, FieldType& rhs, const FEMVector<point_t>& rhsVector)
            : Base(lhs, lhs, rhs)
            , rhsVector_m(nullptr)
            , refElement_m()
            , quadrature_m(refElement_m, 0.0, 0.0)
            , nedelecSpace_m(rhs.get_mesh(), refElement_m, quadrature_m, rhs.getLayout()) {
            
            static_assert(std::is_floating_point<T>::value, "Not a floating point type");
            setDefaultParameters();

            // start a timer
            rhsVector_m =
                //std::make_unique<FEMVector<T>>(nedelecSpace_m.evaluateLoadVectorFunctor(functor));
                std::make_unique<FEMVector<T>>(nedelecSpace_m.evaluateLoadVector(rhsVector));

            rhsVector_m->accumulateHalo();
            rhsVector_m->fillHalo();
            
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

            const Vector<size_t, Dim> zeroNdIndex = Vector<size_t, Dim>(0);

            // We can pass the zeroNdIndex here, since the transformation
            // jacobian does not depend on translation
            const auto firstElementVertexPoints =
                nedelecSpace_m.getElementMeshVertexPoints(zeroNdIndex);

            // Compute Inverse Transpose Transformation Jacobian ()
            const Vector<T, Dim> DPhiInvT =
                refElement_m.getInverseTransposeTransformationJacobian(firstElementVertexPoints);

            // Compute absolute value of the determinant of the transformation
            // jacobian (|det D Phi_K|)
            const T absDetDPhi = Kokkos::abs(
                refElement_m.getDeterminantOfTransformationJacobian(firstElementVertexPoints));

            EvalFunctor<T, Dim, this->nedelecSpace_m.numElementDOFs> maxwellDiffusionEval(
                DPhiInvT, absDetDPhi);

            const auto algoOperator = [maxwellDiffusionEval, this](FEMVector<T> vector)
                                                                            -> FEMVector<T> {

                vector.fillHalo();

                FEMVector<T> return_vector = nedelecSpace_m.evaluateAx(vector,maxwellDiffusionEval);

                return_vector.accumulateHalo();

                return return_vector;
            };

            pcg_algo_m.setOperator(algoOperator);

            
                


            //FEMVector<T> lhsVector = lagrangeSpace_m.interpolateToFEMVector(*(this->lhs_mp));
            FEMVector<T> lhsVector = rhsVector_m->deepCopy();
            lhsVector = 0;
            
            try {
                pcg_algo_m(lhsVector, *rhsVector_m, this->params_m);
            } catch (IpplException& e) {
                std::cout << e.where() << ": " << e.what() << "\n";
                std::cout << "EXITING\n";
                exit(-1);
            }
            
            lhsVector.fillHalo();

            // set the boundary values to the correct values.

            
            //lagrangeSpace_m.reconstructToField(lhsVector, *(this->lhs_mp));
            //nedelecSpace_m.reconstructSolution(lhsVector, *(this->En_mp));
            //lagrangeSpace_m.reconstructToField(*rhsVector_m, *(this->rhs_mp));

            lhsVector_m = std::make_unique<FEMVector<T>>(lhsVector);
            return lhsVector.template skeletonCopy<Vector<T,Dim> >();//nedelecSpace_m.reconstructBasis(lhsVector);
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
         * @brief Reconstructs function values at arbitrary points in the mesh.
         * 
         * This function can be used to retrieve the values of a solution
         * function at arbitrary points inside of the mesh.
         * 
         * @note Currently the function is able to handle both cases, where we
         * have that \p positions only contains positions which are inside of
         * local domain of this MPI rank (i.e. each rank gets its own unique
         * \p position ) and where \p positions contains the positions of all
         * ranks (i.e. \p positions is the same for all ranks). If in the future
         * it can be guaranteed, that each rank will get its own \p positions
         * then certain parts of the function implementation can be removed.
         * Instructions for this are given in the implementation itself.
         * 
         * @param positions The points at which the function should be
         * evaluated. A \c Kokkos::View which stores in each element a 2D/3D 
         * point.
         * 
         * @return The function evaluated at the given points, stored inside of
         * \c Kokkos::View where each element corresponts to the function value
         * at the point described by the same element inside of \p positions.
         */
        Kokkos::View<point_t*> reconstructToPoints(const Kokkos::View<point_t*>& positions) <%
            return this->nedelecSpace_m.reconstructToPoints(positions, *lhsVector_m);
        %>


        
        /**
         * @brief Given an analytical solution computes the L2 norm error.
         *
         * @param analytical The analytical solution (functor)
         *
         * @return error - The error ||u - analytical||_L2
         */
        template <typename F>
        T getL2Error(const F& analytic) {
            T error_norm = this->nedelecSpace_m.computeError(*lhsVector_m, analytic);
            return error_norm;
        }


    protected:
        PCGSolverAlgorithm_t pcg_algo_m;
        

        virtual void setDefaultParameters() {
            this->params_m.add("max_iterations", 10);
            this->params_m.add("tolerance", (T)1e-13);
        }

        std::unique_ptr<FEMVector<T>> rhsVector_m;

        std::unique_ptr<FEMVector<T>> lhsVector_m;

        ElementType refElement_m;
        QuadratureType quadrature_m;
        NedelecType nedelecSpace_m;
    };

}  // namespace ippl



#endif // IPPL_FEM_MAXWELL_DIFFUSION_SOLVER_H