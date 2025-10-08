// Class NedelecSpace
//    This is the NedelecSpace class. It is a class representing a Nedelec space
//    for finite element methods on a structured grid.

#ifndef IPPL_NEDELECSPACE_H
#define IPPL_NEDELECSPACE_H

#include <cmath>

#include "FEM/FiniteElementSpace.h"
#include "FEM/FEMVector.h"

constexpr unsigned getNedelecNumElementDOFs(unsigned Dim, [[maybe_unused]] unsigned Order) {
    // needs to be constexpr pow function to work at compile time. Kokkos::pow
    // doesn't work.
    return static_cast<unsigned>(static_cast<int>(Dim)*power(2, static_cast<int>(Dim-1)));
}

namespace ippl {

    /**
     * @brief A class representing a Nedelec space for finite element methods on
     * a structured, rectilinear grid.
     *
     * @tparam T The floating point number type of the field values
     * @tparam Dim The dimension of the mesh
     * @tparam Order The order of the Nedelec space
     * @tparam QuadratureType The type of the quadrature rule
     * @tparam FieldType The type of field to use.
     */
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    // requires IsQuadrature<QuadratureType>
    class NedelecSpace : public FiniteElementSpace<T, Dim, getNedelecNumElementDOFs(Dim, Order),
            ElementType, QuadratureType, FEMVector<T>, FEMVector<T>> {
    public:
        // The number of degrees of freedom per element
        static constexpr unsigned numElementDOFs = getNedelecNumElementDOFs(Dim, Order);

        // The dimension of the mesh
        static constexpr unsigned dim = FiniteElementSpace<T, Dim, numElementDOFs, ElementType,
                                            QuadratureType, FEMVector<T>, FEMVector<T>>::dim;

        // The order of the Nedelec space
        static constexpr unsigned order = Order;

        // The number of mesh vertices per element
        static constexpr unsigned numElementVertices = FiniteElementSpace<T, Dim, numElementDOFs,
                        ElementType, QuadratureType, FEMVector<T>, FEMVector<T>>::numElementVertices;

        // A vector with the position of the element in the mesh in each dimension
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType,
                                            FEMVector<T>, FEMVector<T>>::indices_t indices_t;

        // A point in the global coordinate system
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType,
                                            FEMVector<T>, FEMVector<T>>::point_t point_t;

        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType,
                                            FEMVector<T>, FEMVector<T>>::vertex_points_t vertex_points_t;

        // Field layout type for domain decomposition info
        typedef FieldLayout<Dim> Layout_t;

        // View types
        typedef typename detail::ViewType<T, 1>::view_type ViewType;
        typedef typename detail::ViewType<T, 1,
                                Kokkos::MemoryTraits<Kokkos::Atomic>>::view_type AtomicViewType;



        ///////////////////////////////////////////////////////////////////////
        // Constructors ///////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Construct a new NedelecSpace object
         *
         * @param mesh Reference to the mesh
         * @param ref_element Reference to the reference element
         * @param quadrature Reference to the quadrature rule
         * @param layout Reference to the field layout
         */
        NedelecSpace(UniformCartesian<T, Dim>& mesh, ElementType& ref_element,
                                    const QuadratureType& quadrature, const Layout_t& layout);

        /**
         * @brief Construct a new NedelecSpace object (without layout)
         * This constructor is made to work with the default constructor in
         * FEMPoissonSolver.h such that it is compatible with alpine.
         *
         * @param mesh Reference to the mesh
         * @param ref_element Reference to the reference element
         * @param quadrature Reference to the quadrature rule
         */
        NedelecSpace(UniformCartesian<T, Dim>& mesh, ElementType& ref_element,
                                    const QuadratureType& quadrature);

        /**
         * @brief Initialize a NedelecSpace object created with the default
         * constructor.
         *
         * @param mesh Reference to the mesh
         * @param layout Reference to the field layout
         */
        void initialize(UniformCartesian<T, Dim>& mesh, const Layout_t& layout);

        ///////////////////////////////////////////////////////////////////////
        /**
         * @brief Initialize a Kokkos view containing the element indices
         */
        void initializeElementIndices(const Layout_t& layout);

        /// Degree of Freedom operations //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Get the number of global degrees of freedom in the space
         *
         * @return size_t - unsigned integer number of global degrees of freedom
         */
        KOKKOS_FUNCTION size_t numGlobalDOFs() const override;

        /**
         * @brief Get the elements local DOF from the element index and global
         * DOF index
         *
         * @param elementIndex size_t - The index of the element
         * @param globalDOFIndex size_t - The global DOF index
         *
         * @return size_t - The local DOF index
         */
        KOKKOS_FUNCTION size_t getLocalDOFIndex(const size_t& elementIndex,
                                    const size_t& globalDOFIndex) const override;

        /**
         * @brief Get the global DOF index from the element index and local DOF.
         *
         * @param elementIndex size_t - The index of the element
         * @param localDOFIndex size_t - The local DOF index
         *
         * @return size_t - The global DOF index
         */
        KOKKOS_FUNCTION size_t getGlobalDOFIndex(const size_t& elementIndex,
                                    const size_t& localDOFIndex) const override;

        /**
         * @brief Get the local DOF indices (vector of local DOF indices)
         * They are independent of the specific element because it only depends
         * on the reference element type.
         *
         * @return Vector<size_t, NumElementDOFs> - The local DOF indices
         */
        KOKKOS_FUNCTION Vector<size_t, numElementDOFs> getLocalDOFIndices() const override;

        /**
         * @brief Get the global DOF indices (vector of global DOF indices) of
         * an element.
         *
         * @param elementIndex size_t - The index of the element
         *
         * @return Vector<size_t, NumElementDOFs> - The global DOF indices
         */
        KOKKOS_FUNCTION Vector<size_t, numElementDOFs> getGlobalDOFIndices(
                                    const size_t& elementIndex) const override;
        
        /**
         * @brief Get the global DOF indices (vector of global DOF indices) of
         * an element.
         *
         * @param elementIndex indices_t - The index of the element
         *
         * @return Vector<size_t, NumElementDOFs> - The global DOF indices
         */
        KOKKOS_FUNCTION Vector<size_t, numElementDOFs> getGlobalDOFIndices(
                                    const indices_t& elementIndex) const;
        
        /**
         * @brief Get the DOF indices (vector of indices) corresponding to the
         * position inside the FEMVector of an element
         *
         * @param elementIndex size_t - The index of the element
         *
         * @return Vector<size_t, NumElementDOFs> - The DOF indices
         */
        KOKKOS_FUNCTION Vector<size_t, numElementDOFs> getFEMVectorDOFIndices(
                                    const size_t& elementIndex, NDIndex<Dim> ldom) const;

        /**
         * @brief Get the DOF indices (vector of indices) corresponding to the
         * position inside the FEMVector of an element
         *
         * @param elementIndex indices_t - The index of the element
         *
         * @return Vector<size_t, NumElementDOFs> - The DOF indices
         */
        KOKKOS_FUNCTION Vector<size_t, numElementDOFs> getFEMVectorDOFIndices(
                                    indices_t elementIndex, NDIndex<Dim> ldom) const;

        
        /**
         * @brief Get the cartesion position of a local DOF in the reference
         * element.
         * 
         * Given the local DOF index this function will return the cartesian
         * position of this DOF with respect to the reference element.
         * 
         */
        KOKKOS_FUNCTION point_t getLocalDOFPosition(size_t localDOFIndex) const;

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////


        /**
         * @brief Evaluate the shape function of a local degree of freedom at a
         * given point in the reference element
         *
         * @param localDOF size_t - The local degree of freedom index
         * @param localPoint point_t (Vector<T, Dim>) - The point in the
         * reference element
         *
         * @return T - The value of the shape function at the given point
         */
        KOKKOS_FUNCTION point_t evaluateRefElementShapeFunction(const size_t& localDOF,
                                    const point_t& localPoint) const;
        

        /**
         * @brief Evaluate the curl of the shape function of a local degree of
         * of freedom at ta given point in the reference element.
         *
         * @param localDOF size_t - The local degree of freedom index
         * @param localPoint point_t (Vector<T, Dim>) - The point in the
         * reference element
         *
         * @return point_t (Vector<T, Dim>) - The curl of the shape function
         * at the given point
         */
        KOKKOS_FUNCTION point_t evaluateRefElementShapeFunctionCurl(const size_t& localDOF,
            const point_t& localPoint) const;
        
        ///////////////////////////////////////////////////////////////////////
        /// Assembly operations ///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Assemble the left stiffness matrix A of the system Ax = b
         *
         * @param x The vector which we want to multiply
         *
         * @return The vector containing A*x
         */
        template <typename F>
        FEMVector<T> evaluateAx(FEMVector<T>& x, F& evalFunction) const;

        /**
         * @brief Assemble the load vector b of the system Ax = b, given a field
         * of the right hand side defined at the Nédélec DOF positions. If a
         * functor instead of a field should be used, use the function 
         * \c NedelecSpace::evaluateLoadVectorFunctor.
         *
         * @param f The source field defined at the Nédélec degrees fo freedom.
         *
         * @return The resulting rhs b of the Galerkin discretization.
         */
        FEMVector<T> evaluateLoadVector(const FEMVector<point_t>& f) const;
        

        /**
         * @brief Assemble the load vector b of the system Ax = b, given a
         * functional of the rhs. If a field instead of a functor should be
         * used, use the function \c NedelecSpace::evaluateLoadVector.
         *
         * @param f The source function, which can be evaluated at arbitrary
         * points.
         * 
         * @tparam F The functor type.
         * 
         * @return The resulting rhs b of the Galerkin discretization.
         */
        template <typename F>
        FEMVector<T> evaluateLoadVectorFunctor(const F& f) const;



        ///////////////////////////////////////////////////////////////////////
        /// FEMVector conversion and creation//////////////////////////////////
        ///////////////////////////////////////////////////////////////////////
        
        /**
         * @brief Creates and empty FEMVector.
         * 
         * Creates and empty FEMVector which corresponds to the domain this MPI
         * rank owns (according to the ippl layout created for this mesh). To
         * this extend it will also setup all the information needed to exchange
         * halo cells.
         * 
         * @returns An empty FEMVector for this domain.
         */
        FEMVector<T> createFEMVector() const;
        

        /**
         * @brief Reconstructs function values at arbitrary points in the mesh
         * given the Nedelec DOF coefficients.
         * 
         * This function can be used to retrieve the values of a solution
         * function at arbitrary points inside of the mesh given the Nedelec
         * DOF coefficients which solved the problem using FEM.
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
         * @param coef The basis function coefficients obtained via FEM.
         * 
         * @return The function evaluated at the given points, stored inside of
         * \c Kokkos::View where each element corresponts to the function value
         * at the point described by the same element inside of \p positions.
         */
        Kokkos::View<point_t*> reconstructToPoints(const Kokkos::View<point_t*>& positions,
            const FEMVector<T>& coef) const;

        ///////////////////////////////////////////////////////////////////////
        /// Error norm computations ///////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////
        

        /**
         * @brief Given the Nedelec space DoF coefficients and an analytical
         * solution computes the L2 norm error.
         *
         * @param u_h The basis function coefficients obtained via FEM.
         * @param u_sol The analytical solution (functor)
         *
         * @return error - The error ||u_h - u_sol||_L2
         */
        template <typename F> T computeError(const FEMVector<T>& u_h, const F& u_sol) const;


        /**
         * @brief Check if a DOF is on the boundary of the mesh
         *
         * This function takes as input the global index of a DoF and returns
         * if this DoF is on. If one would like to know which boundary this is
         * the function \c NedelecSpace::getBoundarySide can be used.
         * 
         * @param dofIdx The global DoF index for which should be checked if it
         * is on the boundary.
         *
         * @return true - If the DOF is on the domain boundary
         * @return false - If the DOF is not on the domain boundary
         */
        KOKKOS_FUNCTION bool isDOFOnBoundary(const size_t& dofIdx) const;

        /** 
        * @brief Returns which side the boundary is on.
        * 
        * This function takes as input the global index of a DoF and then
        * returns on which side of the domain boundary it is on, in 2d that
        * would be either south, north, west, east and in 3d space and ground is
        * added. The mapping is as follows:
        * 0 = south
        * 1 = west
        * 2 = north
        * 3 = east
        * 4 = ground
        * 5 = space
        * -1 = not on a boundary.
        * 
        * @param dofIdx the global DoF index for which the boundary side should
        * be retrieved.
        * 
        * @returns Which boundary side the DoF is on or -1 if on no boundary.
        */
        KOKKOS_FUNCTION int getBoundarySide(const size_t& dofIdx) const;


    
    private:
        /**
         * @brief Implementation of the \c NedelecSpace::createFEMVector
         * function for 2d.
         */
        FEMVector<T> createFEMVector2d() const;

        /**
         * @brief Implementation of the \c NedelecSpace::createFEMVector
         * function for 3d.
         */
        FEMVector<T> createFEMVector3d() const;
        
        /**
         * @brief Stores which elements (squares or cubes) belong to the current
         * MPI rank.
         */
        Kokkos::View<size_t*> elementIndices;

        /**
         * @brief Stores the positions of the local Degrees of Freedoms on the 
         * reference elements.
         * 
         * We are saying that the local degree of freedom positions are simply
         * the centers of the edges. 
         */
        Vector<point_t, 12> localDofPositions_m;
        
        /**
         * @brief The layout of the MPI ranks over the mesh.
         * 
         * Standart ippl layout which dictates how the MPI ranks are layed out
         * over the mesh. It is used in order to be able to create FEMVectors,
         * retreive correct DOF indices and intitalize the elementIndices.
         */
        Layout_t layout_m;

    };

}  // namespace ippl

#include "FEM/NedelecSpace.hpp"

#endif
