// Class LagrangeSpace
//    This is the LagrangeSpace class. It is a class representing a Lagrange space
//    for finite element methods on a structured grid.

#ifndef IPPL_LAGRANGESPACE_H
#define IPPL_LAGRANGESPACE_H

#include <cmath>

#include "FEM/FEMQuadratureData.h"
#include "FEM/FiniteElementSpace.h"

constexpr unsigned getLagrangeNumElementDOFs(unsigned Dim, unsigned Order) {
    // needs to be constexpr pow function to work at compile time. Kokkos::pow doesn't work.
    return static_cast<unsigned>(power(static_cast<int>(Order + 1), static_cast<int>(Dim)));
}

namespace ippl {

    /**
     * @brief A class representing a Lagrange space for finite element methods on a structured,
     * rectilinear grid.
     *
     * @tparam T The floating point number type of the field values
     * @tparam Dim The dimension of the mesh
     * @tparam Order The order of the Lagrange space
     * @tparam QuadratureType The type of the quadrature rule
     * @tparam FieldLHS The type of the left hand side field
     * @tparam FieldRHS The type of the right hand side field
     */
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    // requires IsQuadrature<QuadratureType>
    class LagrangeSpace
        : public FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order), ElementType,
                                    QuadratureType, FieldLHS, FieldRHS> {
    public:
        // The number of degrees of freedom per element
        static constexpr unsigned numElementDOFs = getLagrangeNumElementDOFs(Dim, Order);

        // The dimension of the mesh
        static constexpr unsigned dim = FiniteElementSpace<T, Dim, numElementDOFs, ElementType,
                                                           QuadratureType, FieldLHS, FieldRHS>::dim;

        // The order of the Lagrange space
        static constexpr unsigned order = Order;

        // The number of mesh vertices per element
        static constexpr unsigned numElementVertices =
            FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType, FieldLHS,
                               FieldRHS>::numElementVertices;

        // A vector with the position of the element in the mesh in each dimension
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType,
                                            FieldLHS, FieldRHS>::indices_t indices_t;

        // A point in the global coordinate system
        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType,
                                            FieldLHS, FieldRHS>::point_t point_t;

        typedef typename FiniteElementSpace<T, Dim, numElementDOFs, ElementType, QuadratureType,
                                            FieldLHS, FieldRHS>::vertex_points_t vertex_points_t;

        // Field layout type for domain decomposition info
        typedef FieldLayout<Dim> Layout_t;

        // View types
        typedef typename detail::ViewType<T, Dim>::view_type ViewType;
        typedef typename detail::ViewType<T, Dim, Kokkos::MemoryTraits<Kokkos::Atomic>>::view_type
            AtomicViewType;

        ///////////////////////////////////////////////////////////////////////
        // Constructors ///////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Construct a new LagrangeSpace object
         *
         * @param mesh Reference to the mesh
         * @param ref_element Reference to the reference element
         * @param quadrature Reference to the quadrature rule
         * @param layout Reference to the field layout
         */
        LagrangeSpace(UniformCartesian<T, Dim>& mesh, ElementType& ref_element,
                      const QuadratureType& quadrature, Layout_t& layout);

        /**
         * @brief Construct a new LagrangeSpace object (without layout)
         * This constructor is made to work with the default constructor in
         * FEMPoissonSolver.h such that it is compatible with alpine.
         *
         * @param mesh Reference to the mesh
         * @param ref_element Reference to the reference element
         * @param quadrature Reference to the quadrature rule
         */
        LagrangeSpace(UniformCartesian<T, Dim>& mesh, ElementType& ref_element,
                      const QuadratureType& quadrature);

        /**
         * @brief Initialize a LagrangeSpace object created with the default constructor
         *
         * @param mesh Reference to the mesh
         * @param layout Reference to the field layout
         */
        void initialize(UniformCartesian<T, Dim>& mesh, Layout_t& layout);

        ///////////////////////////////////////////////////////////////////////
        /**
         * @brief Initialize a Kokkos view containing the element indices.
         * This distributes the elements among MPI ranks.
         */
        void initializeElementIndices(Layout_t& layout);

        ///////////////////////////////////////////////////////////////////////
        /**
         * @brief Function to update the element partition and the layout of
         * fields in the LagrangeSpace if the layout has been changed during
         * the simulation (for example by the load balancer).
         */
        void updateLayout(Layout_t& layout);

        /// Degree of Freedom operations //////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Get the number of global degrees of freedom in the space
         *
         * @return size_t - unsigned integer number of global degrees of freedom
         */
        KOKKOS_FUNCTION size_t numGlobalDOFs() const override;

        /**
         * @brief Get the elements local DOF from the element index and global DOF
         * index
         *
         * @param elementIndex size_t - The index of the element
         * @param globalDOFIndex size_t - The global DOF index
         *
         * @return size_t - The local DOF index
         */
        KOKKOS_FUNCTION size_t getLocalDOFIndex(const size_t& elementIndex,
                                                const size_t& globalDOFIndex) const override;

        /**
         * @brief Get the global DOF index from the element index and local DOF
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
         * They are independent of the specific element because it only depends on
         * the reference element type
         *
         * @return Vector<size_t, NumElementDOFs> - The local DOF indices
         */
        KOKKOS_FUNCTION Vector<size_t, numElementDOFs> getLocalDOFIndices() const override;

        /**
         * @brief Get the global DOF indices (vector of global DOF indices) of an element
         *
         * @param element_index size_t - The index of the element
         *
         * @return Vector<size_t, NumElementDOFs> - The global DOF indices
         */
        KOKKOS_FUNCTION Vector<size_t, numElementDOFs> getGlobalDOFIndices(
            const size_t& element_index) const override;

        ///////////////////////////////////////////////////////////////////////
        /// Basis functions and gradients /////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Evaluate the shape function of a local degree of freedom at a given point in the
         * reference element
         *
         * @param localDOF size_t - The local degree of freedom index
         * @param localPoint point_t (Vector<T, Dim>) - The point in the reference element
         *
         * @return T - The value of the shape function at the given point
         */
        KOKKOS_FUNCTION T evaluateRefElementShapeFunction(const size_t& localDOF,
                                                          const point_t& localPoint) const;

        /**
         * @brief Evaluate the gradient of the shape function of a local degree of freedom at a
         * given point in the reference element
         *
         * @param localDOF size_t - The local degree of freedom index
         * @param localPoint point_t (Vector<T, Dim>) - The point in the reference element
         *
         * @return point_t (Vector<T, Dim>) - The gradient of the shape function at the given
         * point
         */
        KOKKOS_FUNCTION point_t evaluateRefElementShapeFunctionGradient(
            const size_t& localDOF, const point_t& localPoint) const;

        ///////////////////////////////////////////////////////////////////////
        /// Functions to access element info from outside /////////////////////
        ///////////////////////////////////////////////////////////////////////

        KOKKOS_FUNCTION point_t
        getInverseTransposeTransformationJacobian(vertex_points_t pt) const {
            return this->ref_element_m.getInverseTransposeTransformationJacobian(pt);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Assembly operations ///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Assemble the left stiffness matrix A of the system Ax = b
         *
         * @param field The field to assemble the matrix for
         * @param evalFunction The lambda telling us the form which A takes
         *
         * @return FieldLHS - The LHS field containing A*x
         */
        template <typename F>
        FieldLHS evaluateAx(FieldLHS& field, F& evalFunction);

        template <typename F>
        FieldLHS evaluateAx_lower(FieldLHS& field, F& evalFunction);

        template <typename F>
        FieldLHS evaluateAx_upper(FieldLHS& field, F& evalFunction);

        template <typename F>
        FieldLHS evaluateAx_upperlower(FieldLHS& field, F& evalFunction);

        template <typename F>
        FieldLHS evaluateAx_inversediag(FieldLHS& field, F& evalFunction);

        template <typename F>
        FieldLHS evaluateAx_diag(FieldLHS& field, F& evalFunction);

        /**
         * @brief Assemble the left stiffness matrix A of the system
         * but only for the boundary values, so that they can be
         * subtracted from the RHS for treatment of Dirichlet BCs
         *
         * @param field The field to assemble the matrix for
         * @param evalFunction The lambda telling us the form which A takes
         *
         * @return FieldLHS - The LHS field containing A*x
         */
        template <typename F>
        FieldLHS evaluateAx_lift(FieldLHS& field, F& evalFunction);

        /**
         * @brief Assemble the load vector b of the system Ax = b
         *
         * @param field The field to set with the load vector
         */
        void evaluateLoadVector(FieldRHS& field) const;
        void evaluateLumpedMass(FieldRHS& field) const;

        ///////////////////////////////////////////////////////////////////////
        /// Error norm computations ///////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Given two fields, compute the L2 norm error
         *
         * @param u_h The numerical solution found using FEM
         * @param u_sol The analytical solution (functor)
         *
         * @return error - The error ||u_h - u_sol||_L2
         */
        template <typename F>
        T computeErrorL2(const FieldLHS& u_h, const F& u_sol) const;

        /**
         * @brief Given a field, compute the average
         *
         * @param u_h The numerical solution found using FEM
         *
         * @return avg The average of the field on the domain
         */
        T computeAvg(const FieldLHS& u_h) const;

        ///////////////////////////////////////////////////////////////////////
        /// Device struct for copies //////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////
        /**
         * @brief Device-copyable snapshot of the Lagrange-space geometry and indexing rules.
         *
         * @details
         * `LagrangeSpace` is a host-side owner. In particular, it contains fields and other
         * objects whose lifetime management and destructors are not device-callable. Capturing
         * the complete space (or `this`) in a `KOKKOS_CLASS_LAMBDA` therefore makes the kernel
         * closure contain host-only state. CUDA compilers can then diagnose calls to host-only
         * constructors or destructors while generating device code.
         *
         * `DeviceStruct` defines the architectural boundary between that host-side owner and
         * device kernels. It contains only the immutable, device-copyable values needed to
         * reproduce mesh indexing, element geometry, degree-of-freedom mappings, and reference
         * element evaluations. A kernel obtains a snapshot with getDeviceMirror() and captures
         * that snapshot by value in a `KOKKOS_LAMBDA`. Kokkos views such as `elementIndices` and
         * field data views are captured separately; they deliberately are not owned by this
         * structure.
         *
         * More specifically, the mirror performs the following tasks:
         *
         * - It snapshots the mesh vertex counts (`nr_m`), mesh spacing (`hr_m`), physical origin
         *   (`origin_m`), and reference element (`ref_element_m`) that device code needs for
         *   geometric calculations. The compile-time element/DOF counts are reused without
         *   adding runtime storage.
         * - It reconstructs the deterministic operations that previously required access to the
         *   parent class: flattened/N-dimensional element-index conversion, element-vertex
         *   enumeration, physical vertex-coordinate construction, global-DOF lookup, boundary
         *   detection, and reference-element shape-function evaluation.
         * - It gives a `KOKKOS_LAMBDA` a self-contained, read-only description of the space. The
         *   host creates the structure before launching a kernel; capture-by-value places it in
         *   the Kokkos kernel closure, which Kokkos makes available in the selected execution
         *   space. getDeviceMirror() itself performs no field allocation or field-data copy.
         * - It does not contain field objects, field layouts, MPI decomposition state, or other
         *   host-owned infrastructure. Before each kernel launch, the host extracts the required
         *   device-accessible Kokkos views—such as the `resultField` data view and
         *   `elementIndices`—and captures those view handles separately from this geometry and
         *   indexing snapshot.
         * - It removes the need to capture `this`. Consequently, constructing and destroying the
         *   device closure never requires the host-only lifetime operations of `LagrangeSpace` or
         *   `Field`, which would cause issues when using GPUs.
         *
         * "Lightweight" therefore means that this is neither a second owning finite-element space
         * nor an automatically synchronized copy of one. It is a small, non-owning value snapshot
         * containing only kernel-invariant geometry and algorithms. Changes to the host object are
         * not reflected in an existing mirror; the host must create another mirror before the next
         * kernel that observes the changed state.
         *
         * When a new device kernel needs additional `LagrangeSpace` functionality, add the
         * smallest required device-copyable state and a `KOKKOS_FUNCTION` helper here instead of
         * capturing the parent object. All members must remain safe to copy into a device closure
         * and must not introduce host-only ownership or lifetime management. A new snapshot must
         * be created after changing any mirrored mesh or reference-element state on the host.
         * During a kernel invocation the snapshot is read-only and may be shared by all threads.
         *
         */
        struct DeviceStruct {
            // members we need to copy for the following functions:
            // works since numElementDOFs in LagrangeSpace is static constexpr
            static constexpr unsigned numElementDOFs     = LagrangeSpace::numElementDOFs;
            static constexpr unsigned numElementVertices = LagrangeSpace::numElementVertices;
            using indices_list_t                         = Vector<indices_t, numElementVertices>;
            using vertex_points_t                        = Vector<point_t, numElementVertices>;

            Vector<size_t, Dim> nr_m;      ///< Number of mesh vertices in each dimension.
            Vector<double, Dim> hr_m;      ///< Uniform mesh spacing in each dimension.
            Vector<double, Dim> origin_m;  ///< Physical coordinate of mesh vertex index zero.
            ElementType ref_element_m;     ///< Device-copyable reference-element description.

            // these are the functions needed for interpolation to the space
            KOKKOS_FUNCTION indices_t getMeshVertexNDIndex(const size_t& vertex_index) const;

            /**
             * @brief Convert a flattened element index to its N-dimensional mesh index.
             *
             * The returned index denotes the lower mesh vertex of the element. Dimension zero
             * is the fastest-varying dimension in the flattened representation.
             *
             * @param element_index Zero-based flattened element index in the interval
             *        `[0, product(nr_m[d] - 1))`.
             * @return N-dimensional element index in which component `d` is in the interval
             *         `[0, nr_m[d] - 1)`.
             *
             * @pre Every dimension contains at least two mesh vertices.
             * @see getElementIndex()
             */
            KOKKOS_FUNCTION indices_t getElementNDIndex(const size_t& element_index) const;

            /**
             * @brief Flatten an N-dimensional element index.
             *
             * This is the inverse of getElementNDIndex() for valid element indices and uses a
             * dimension-zero-fastest ordering.
             *
             * @param element_nd_index N-dimensional index of an element's lower mesh vertex.
             *        Each component `d` must be in `[0, nr_m[d] - 1)`.
             * @return Zero-based flattened element index.
             *
             * @see getElementNDIndex()
             */
            KOKKOS_FUNCTION size_t getElementIndex(const indices_t& element_nd_index) const;

            /**
             * @brief Return the mesh indices of every vertex belonging to an element.
             *
             * Vertex ordering follows the tensor-product binary convention used throughout the
             * finite-element implementation: bit `d` of a local vertex number selects the lower
             * (`0`) or upper (`1`) vertex in dimension `d`.
             *
             * @param element_nd_index N-dimensional index of the element's lower mesh vertex.
             * @return Fixed-size list of the element vertex indices in local vertex order.
             */
            KOKKOS_FUNCTION indices_list_t
            getElementMeshVertexNDIndices(const indices_t& element_nd_index) const;

            /**
             * @brief Return the physical coordinates of every vertex belonging to an element.
             *
             * Coordinates are computed from the mirrored structured-mesh geometry as
             * `origin_m[d] + vertex_index[d] * hr_m[d]`. The returned points use the same local
             * vertex ordering as getElementMeshVertexNDIndices().
             *
             * @param element_nd_index N-dimensional index of the element's lower mesh vertex.
             * @return Fixed-size list of physical vertex coordinates in local vertex order.
             *
             * @see getElementMeshVertexNDIndices()
             */
            KOKKOS_FUNCTION vertex_points_t
            getElementMeshVertexPoints(const indices_t& element_nd_index) const;

            KOKKOS_FUNCTION size_t getLocalDOFIndex(const indices_t& elementNDIndex,
                                                    const size_t& globalDOFIndex) const;

            /**
             * @brief Return the global degree-of-freedom indices of a flattened element.
             *
             * This convenience overload first converts the flattened element index with
             * getElementNDIndex() and then applies the existing N-dimensional DOF mapping. It
             * allows kernels that iterate over the flattened `elementIndices` view to remain
             * entirely within the device-safe interface.
             *
             * @param elementIndex Zero-based flattened element index.
             * @return Global DOF indices in local element-DOF order.
             *
             * @see getElementNDIndex()
             * @see getGlobalDOFIndices(const indices_t&) const
             */
            KOKKOS_FUNCTION Vector<size_t, numElementDOFs> getGlobalDOFIndices(
                const size_t& elementIndex) const;
            KOKKOS_FUNCTION Vector<size_t, numElementDOFs> getGlobalDOFIndices(
                const indices_t& elementNDIndex) const;

            /**
             * @brief Determine whether a global degree of freedom lies on the mesh boundary.
             *
             * A DOF is a boundary DOF when at least one index component is on the lower boundary
             * (`0`) or upper boundary (`nr_m[d] - 1`). The predicate is used inside assembly
             * kernels to apply or skip Dirichlet boundary contributions without accessing the
             * host-side `LagrangeSpace` object.
             *
             * @param ndindex N-dimensional global mesh/DOF index.
             * @return `true` if any component lies on a domain boundary; otherwise `false`.
             */
            KOKKOS_FUNCTION bool isDOFOnBoundary(const indices_t& ndindex) const;

            KOKKOS_FUNCTION T evaluateRefElementShapeFunction(const size_t& localDOF,
                                                              const point_t& localPoint) const;
            KOKKOS_FUNCTION point_t evaluateRefElementShapeFunctionGradient(
                const size_t& localDOF, const point_t& localPoint) const;
        };

        /**
         * @brief Create the device-safe snapshot captured by LagrangeSpace kernels.
         *
         * Copies the mesh extents, spacing, origin, and reference element into a non-owning value
         * that can be captured by `KOKKOS_LAMBDA`. The returned value supplies device-side mesh
         * indexing, physical-coordinate, DOF-mapping, boundary, and reference-element operations
         * without retaining a pointer or reference to the parent `LagrangeSpace`. Views containing
         * the element partition and field data are intentionally captured separately by each
         * kernel.
         *
         * @return Independent, device-copyable snapshot of the current geometric and indexing
         *         state.
         *
         * @note Recreate the mirror after changing the corresponding host-side mesh or reference
         *       element state. The returned object does not synchronize later host changes.
         */
        DeviceStruct getDeviceMirror() const;

    private:
        /**
         * @brief Check if a DOF is on the boundary of the mesh
         *
         * @param ndindex The NDIndex of the global DOF
         *
         * @return true - If the DOF is on the boundary
         * @return false - If the DOF is not on the boundary
         */
        KOKKOS_FUNCTION bool isDOFOnBoundary(const indices_t& ndindex) const {
            for (size_t d = 0; d < Dim; ++d) {
                if (ndindex[d] <= 0 || ndindex[d] >= this->nr_m[d] - 1) {
                    return true;
                }
            }
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        /// Private member containing the element indices owned by ////////////
        /// my MPI rank. //////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////
        Kokkos::View<size_t*> elementIndices;

        // One time allocated field of type FieldLHS to store results
        FieldLHS resultField;
    };

}  // namespace ippl

#include "FEM/LagrangeSpace.hpp"

#endif
