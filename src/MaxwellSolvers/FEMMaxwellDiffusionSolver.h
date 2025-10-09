// Class FEMMaxwellDifussionSolver
//   Solves the electric diffusion probelm given by curl(curl(E)) + E = f in 
//   the domain and n x E = 0 on the boundary.

#ifndef IPPL_FEM_MAXWELL_DIFFUSION_SOLVER_H
#define IPPL_FEM_MAXWELL_DIFFUSION_SOLVER_H

#include "LinearSolvers/PCG.h"
#include "Maxwell.h"
#include <iomanip>
#include <iostream>
#include <fstream>

namespace ippl {

    /**
     * @brief Representation of the lhs of the problem we are trying to solve.
     * 
     * In our case this corresponds to the variational formulation of the 
     * curl(curl(E)) + E and is curl(b_i)*curl(b_j) + b_i*b_j.
     * 
     * @tparam T The type we are working with.
     * @tparam Dim the dimension of the space.
     * @tparam numElementDOFs the number of DOFs per element that we have.
     */
    template <typename T, unsigned Dim, unsigned numElementDOFs>
    struct EvalFunctor {
        /**
         * @brief The inverse transpose Jacobian.
         * 
         * As we have a unirectangular grid it is the same for all the differnt
         * Elements and we therefore have to store it only once.
         */
        const Vector<T, Dim> DPhiInvT;

        /**
         * @brief The determinant of the Jacobian.
         * 
         * As we have a unirectangular grid it is the same for all the differnt
         * Elements and we therefore have to store it only once.
         */
        const T absDetDPhi;

        /**
         * @brief Constructor.
         */
        EvalFunctor(Vector<T, Dim> DPhiInvT, T absDetDPhi)
            : DPhiInvT(DPhiInvT)
            , absDetDPhi(absDetDPhi) {}

        /**
         * @brief Returns the evaluation of
         * (curl(b_i)*curl(b_j) + b_i*b_j)*absDetDPhi.
         * 
         * This function takes as input the basis function values and their curl
         * for the different DOFs and returns the evaluation of the inner part
         * of the integral of the variational formuation, which corresponds to
         * (curl(b_i)*curl(b_j) + b_i*b_j), but note that we additionally also
         * multiply this with absDetDPhi, which is required by the quadrature
         * rule. In theroy this could also be done outside of this.
         * 
         * @param i The first DOF index.
         * @param j The second DOF index.
         * @param curl_b_q_k The curl of the DOFs.
         * @param val_b_q_k The values of the DOFs.
         * 
         * @returns (curl(b_i)*curl(b_j) + b_i*b_j)*absDetDPhi
         */
        KOKKOS_FUNCTION auto operator()(size_t i, size_t j,
            const ippl::Vector<ippl::Vector<T, Dim>, numElementDOFs>& curl_b_q_k,
            const ippl::Vector<ippl::Vector<T, Dim>, numElementDOFs>& val_b_q_k) const {
            
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

            // Calcualte the rhs, using the Nedelec space
            rhsVector_m =
                std::make_unique<FEMVector<T>>(nedelecSpace_m.evaluateLoadVector(rhsVector));

            rhsVector_m->accumulateHalo();
            rhsVector_m->fillHalo();
            
        }

        void setRhs(FieldType& rhs, const FEMVector<point_t>& rhsVector) {
            
            Base::setRhs(rhs);

            // Calcualte the rhs, using the Nedelec space
            rhsVector_m =
                std::make_unique<FEMVector<T>>(nedelecSpace_m.evaluateLoadVector(rhsVector));

            rhsVector_m->accumulateHalo();
            rhsVector_m->fillHalo();
        }

        /**
         * @brief Solve the equation using finite element methods.
         */
        void solve() override {

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

            // Create the functor object which stores the function we have to
            // solve for the lhs
            EvalFunctor<T, Dim, NedelecType::numElementDOFs> maxwellDiffusionEval(
                DPhiInvT, absDetDPhi);
            
            // The Ax operator
            const auto algoOperator = [maxwellDiffusionEval, this](FEMVector<T> vector)
                                                                            -> FEMVector<T> {

                vector.fillHalo();

                FEMVector<T> return_vector = nedelecSpace_m.evaluateAx(vector,maxwellDiffusionEval);

                return_vector.accumulateHalo();

                return return_vector;
            };

            // setup the CG solver
            pcg_algo_m.setOperator(algoOperator);
            
            // Create the coefficient vector for the solution
            FEMVector<T> lhsVector = rhsVector_m->deepCopy();
            
            // Solve the system using CG
            try {
                pcg_algo_m(lhsVector, *rhsVector_m, this->params_m);
            } catch (IpplException& e) {
                std::string msg = e.where() + ": " + e.what() + "\n";
                Kokkos::abort(msg.c_str());
            }
            
            // store solution.
            lhsVector_m = std::make_unique<FEMVector<T>>(lhsVector);

            // set the boundary values to the correct values.
            lhsVector.fillHalo();

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
        Kokkos::View<point_t*> reconstructToPoints(const Kokkos::View<point_t*>& positions) {
            return this->nedelecSpace_m.reconstructToPoints(positions, *lhsVector_m);
        }


        
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

        /**
         * @brief The CG Solver we use
         */
        PCGSolverAlgorithm_t pcg_algo_m;
        
        /**
         * @brief Sets the default values for the CG solver.
         * Defaults are: max Iterations = 10, tolerance = 1e-13
         */
        virtual void setDefaultParameters() {
            this->params_m.add("max_iterations", 10);
            this->params_m.add("tolerance", (T)1e-13);
        }

        /**
         * @brief FEM represenation of the rhs
         * We use this to store the rhs b of the System Ax = b used in the
         * Galerkin FEM scheme.
         */
        std::unique_ptr<FEMVector<T>> rhsVector_m;

        /**
         * @brief FEM represenation of the solution vector
         * We use this to store the solution x of the System Ax = b used in the
         * Galerkin FEM scheme.
         */
        std::unique_ptr<FEMVector<T>> lhsVector_m;

        /**
         * @brief the reference element we have.
         */
        ElementType refElement_m;

        /**
         * @brief The quadrature rule we use.
         */
        QuadratureType quadrature_m;

        /**
         * @brief The Nedelec Space object.
         * 
         * This is the representation of the Nedelec space that we have and
         * which we use to interact with all the Nedelec stuff.
         */
        NedelecType nedelecSpace_m;
    };

}  // namespace ippl



#endif // IPPL_FEM_MAXWELL_DIFFUSION_SOLVER_H
