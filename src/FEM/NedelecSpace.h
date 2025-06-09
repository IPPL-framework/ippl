// Class NedelecSpace
//    This is the NedelecSpace class. It is a class representing a Nedelec space
//    for finite element methods on a structured grid.

#ifndef IPPL_NEDELECSPACE_H
#define IPPL_NEDELECSPACE_H

#include <cmath>

#include "FEM/FiniteElementSpace.h"
#include "FEM/FEMVector.h"

constexpr unsigned getNedelecNumElementDOFs(unsigned Dim, unsigned Order) {
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
                                    const size_t& element_index) const override;

        
        /**
         * @brief Get the DOF indices (vector of indices) corresponding to the
         * position inside the FEMVector of an element
         *
         * @param elementIndex size_t - The index of the element
         *
         * @return Vector<size_t, NumElementDOFs> - The DOF indices
         */
        KOKKOS_FUNCTION Vector<size_t, numElementDOFs> getFEMVectorDOFIndices(
                                    const size_t& element_index, NDIndex<Dim> ldom) const;

        
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
         * @brief Functor for evaluating a the bilinear form of
         * $(\nabla \times b_j, \nabla \times b_i)$
         * 
         * @return std::function<T(size_t,size_t,size_t)> - The corresponding
         * functor object
         */
        KOKKOS_FUNCTION std::function<T(size_t,size_t,size_t)> curlCurlOperator() const;


        /**
         * @brief Functor for evaluating a the bilinear form of $(b_j, b_i)$
         * 
         * @return std::function<T(size_t,size_t,size_t)> - The corresponding
         * functor object
         */
        KOKKOS_FUNCTION std::function<T(size_t,size_t,size_t)> massOperator() const;


        /**
         * @brief Functor for evaluating a the linear form of $(b_i, f(x))$
         * 
         * @param f function like T(point_t) - The load function object
         * @return std::function<T(size_t,size_t,point_t)> - The corresponding
         * functor object
         */
        template<typename Functor>
        KOKKOS_FUNCTION std::function<T(size_t,size_t, Vector<T,Dim>)>
                                    loadOperator(Functor f) const;


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
         * @brief Evaluate the gradient of the shape function of a local degree
         * of freedom at a given point in the reference element
         *
         * @param localDOF size_t - The local degree of freedom index
         * @param localPoint point_t (Vector<T, Dim>) - The point in the
         * reference element
         *
         * @return point_t (Vector<T, Dim>) - The gradient of the shape function
         * at the given point
         */
        KOKKOS_FUNCTION point_t evaluateRefElementShapeFunctionGradient(const size_t& localDOF,
                                    const point_t& localPoint) const override;
        

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
        FEMVector<T> evaluateLoadVectorFunctor(const FEMVector<NedelecSpace<T, Dim, Order, ElementType,
            QuadratureType, FieldType>::point_t>& model, const F& f) const;



        ///////////////////////////////////////////////////////////////////////
        /// FEMVector conversion //////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Interpolates data from a field to a \c ippl::FEMVector.
         * 
         * Given a field defined on the vertices of a mesh (e.g. ippl::Field)
         * will interpolate the values to the Nédélec DOF positions and return
         * a \c ippl::FEMVector with the appropriate values.
         * 
         * @param field The field from which to interpolate to the
         * \c ippl::FEMVector
         * 
         * @return A \c ippl::FEMVector holding the interpolated data of
         * \p field at the appropriate DOF positions.
         */
        FEMVector<T> interpolateToFEMVector(const FieldType& field) const;

        
        /**
         * @brief Reconstruct a solution given the basis function coefficient
         * vector \p x.
         * 
         * Given the basis function coefficient vector \p x for the Nédélec
         * basis functions this function returns the values of the corresponding
         * field at the mesh vertices.
         * 
         * @param x The coefficient vector.
         * @param field The field to which the solution should be written to.
         * 
         */
        void reconstructSolution(const FEMVector<T>& x, FieldType& field) const;

        ///////////////////////////////////////////////////////////////////////
        /// Error norm computations ///////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Given two fields, compute the L2 norm error
         *
         * @param u_h The numerical solution found using FEM
         * @param u_sol The analytical solution (functor)
         *
         * @return error - The error ||u_h - u_sol||_L2
         */
        template <typename F> T computeError(const FEMVector<Vector<T,Dim> >& u_h, const F& u_sol) const;
        

        /**
         * @brief Given two fields, compute the L2 norm error
         *
         * @param u_h The numerical solution found using FEM
         * @param u_sol The analytical solution (functor)
         *
         * @return error - The error ||u_h - u_sol||_L2
         */
        template <typename F> T computeErrorCoeff(const FEMVector<T>& u_h, const F& u_sol) const;


        /**
         * @brief Check if a DOF is on the boundary of the mesh
         *
         * @param ndindex The NDIndex of the DOF
         *
         * @return true - If the DOF is on the boundary
         * @return false - If the DOF is not on the boundary
         */
        KOKKOS_FUNCTION bool isDOFOnBoundary(const size_t& dofIdx) const {
            
            bool onBoundary = false;
            if constexpr (Dim == 2) {
                size_t nx = this->nr_m[0];
                size_t ny = this->nr_m[1];
                // South
                bool sVal = (dofIdx < nx -1);
                onBoundary = onBoundary || sVal;
                // North
                onBoundary = onBoundary || (dofIdx > nx*(ny-1) + ny*(nx-1) - nx);
                // West
                onBoundary = onBoundary || ((dofIdx >= nx-1) && (dofIdx - (nx-1)) % (2*nx - 1) == 0);
                // East
                onBoundary = onBoundary || ((dofIdx >= 2*nx-2) && ((dofIdx - 2*nx + 2) % (2*nx - 1) == 0));    
            }

            if constexpr (Dim == 3) {
                size_t nx = this->nr_m[0];
                size_t ny = this->nr_m[1];
                size_t nz = this->nr_m[2];

                size_t zOffset = dofIdx / (nx*(ny-1) + ny*(nx-1) + nx*ny);


                if (dofIdx - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset >= (nx*(ny-1) + ny*(nx-1))) {
                    // we are parallel to z axis
                    // therefore we have halve a cell offset and can never be on the ground or in
                    // s
                    size_t f = dofIdx - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset
                        - (nx*(ny-1) + ny*(nx-1));
                    
                    size_t yOffset = f / nx;
                    // South
                    onBoundary = onBoundary || yOffset == 0;
                    // North
                    onBoundary = onBoundary || yOffset == ny-1;

                    size_t xOffset = f % nx;
                    // West
                    onBoundary = onBoundary || xOffset == 0;
                    // East
                    onBoundary = onBoundary || xOffset == nx-1;

                } else {
                    // are parallel to one of the other axes
                    // Ground
                    onBoundary = onBoundary || zOffset == 0;
                    // Space
                    onBoundary = onBoundary || zOffset == nz-1;
                    
                    size_t f = dofIdx - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset;
                    size_t yOffset = f / (2*nx - 1);
                    size_t xOffset = f - (2*nx - 1)*yOffset;

                    if (xOffset < (nx-1)) {
                        // we are parallel to the x axis, therefore we cannot
                        // be on an west or east boundary, but we still can
                        // be on a north or south boundary
                        
                        // South
                        onBoundary = onBoundary || yOffset == 0;
                        // North
                        onBoundary = onBoundary || yOffset == ny-1;
                        
                    } else {
                        // we are parallel to the y axis, therefore we cannot be
                        // on a south or north boundary, but we still can be on
                        // a west or east boundary
                        if (xOffset >= nx-1) {
                            xOffset -= (nx-1);
                        }

                        // West
                        onBoundary = onBoundary || xOffset == 0;
                        // East
                        onBoundary = onBoundary || xOffset == nx-1;
                    }
                }
            }

            return onBoundary;
        }

        /** 
        * @param boundarySide Which boundary we are on, west,... east,...
        * mapping is: 0 = south
        *             1 = west
        *             2 = north
        *             3 = east
        *             4 = ground
        *             5 = space
        */
        KOKKOS_FUNCTION int getBoundarySide(const size_t& dofIdx) const {

            if constexpr (Dim == 2) {
                size_t nx = this->nr_m[0];
                size_t ny = this->nr_m[1];

                // South
                if (dofIdx < nx -1) return 0;
                // West
                if ((dofIdx - (nx-1)) % (2*nx - 1) == 0) return 1;
                // North
                if (dofIdx > nx*(ny-1) + ny*(nx-1) - nx) return 2;
                // East
                if ((dofIdx >= 2*nx-2) && (dofIdx - 2*nx + 2) % (2*nx - 1) == 0) return 3;

                return -1;
            }

            if constexpr (Dim == 3) {
                size_t nx = this->nr_m[0];
                size_t ny = this->nr_m[1];
                size_t nz = this->nr_m[2];

                size_t zOffset = dofIdx / (nx*(ny-1) + ny*(nx-1) + nx*ny);


                if (dofIdx - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset >= (nx*(ny-1) + ny*(nx-1))) {
                    // we are parallel to z axis
                    // therefore we have halve a cell offset and can never be on the ground or in
                    // s
                    size_t f = dofIdx - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset
                        - (nx*(ny-1) + ny*(nx-1));
                    
                    size_t yOffset = f / nx;
                    // South
                    return 0;
                    // North
                    return 2;

                    size_t xOffset = f % nx;
                    // West
                    return 1;
                    // East
                    return 3;

                } else {
                    // are parallel to one of the other axes
                    // Ground
                    return 4;
                    // Space
                    return 5;
                    
                    size_t f = dofIdx - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset;
                    size_t yOffset = f / (2*nx - 1);
                    size_t xOffset = f - (2*nx - 1)*yOffset;

                    if (xOffset < (nx-1)) {
                        // we are parallel to the x axis, therefore we cannot
                        // be on an west or east boundary, but we still can
                        // be on a north or south boundary
                        
                        // South
                        return 0;
                        // North
                        return 2;
                        
                    } else {
                        // we are parallel to the y axis, therefore we cannot be
                        // on a south or north boundary, but we still can be on
                        // a west or east boundary
                        if (xOffset >= nx-1) {
                            xOffset -= (nx-1);
                        }

                        // West
                        return 1;
                        // East
                        return 3;
                    }
                }
                return -1;
            }

        }


    
    private:

        Kokkos::View<size_t*> elementIndices;

        Layout_t layout_m;

    };

}  // namespace ippl

#include "FEM/NedelecSpace.hpp"

#endif
