
namespace ippl {
    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    LagrangeSpace<T, Dim, Order, QuadratureType>::LagrangeSpace(
        const Mesh<T, Dim>& mesh,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::ElementType& ref_element,
        const QuadratureType& quadrature)
        : FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order),
                             getLagrangeNumGlobalDOFs(Dim, Order), QuadratureType>(
            mesh, ref_element, quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    void LagrangeSpace<T, Dim, Order, QuadratureType>::evaluateAx(
        const Vector<T, numGlobalDOFs>& x, Vector<T, numGlobalDOFs>& resultAx) const {
        const std::size_t& numElementDOFs = this->numElementDOFs;

        const std::size_t numElementIntegrationPoints =
            this->quadrature_m.numElementIntegrationPoints();

        // List of quadrature weights
        auto w = this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        auto q = this->quadrature_m.getIntegrationNodesForRefElement();

        // Inverse Transpose Transformation Jacobian
        Vector<T, Dim> DPhiInvT = this->ref_element_m.getInverseTransposeTransformationJacobian(
            this->getElementMeshVertices(0));

        // Absolute value of det Phi_K
        T absDetDPhi = std::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertices(0)));

        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<gradient_vec_t, numElementDOFs>, numElementIntegrationPoints> grad_b_q;
        for (index_t k = 0; k < numElementIntegrationPoints; ++k) {
            for (index_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementBasisGradient(i, q[k]);
            }
        }

        // Allocate memory for the element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        for (index_t element_index = 0; element_index < this->numElements; ++element_index) {
            Vector<index_t, numElementDOFs> global_dofs;
            Vector<index_t, numElementDOFs> local_dofs;

            // 1. Compute the Galerkin element matrix A_K
            for (index_t i = 0; i < numElementDOFs; ++i) {
                for (index_t j = 0; j < numElementDOFs; ++j) {
                    A_K[i][j] = 0.0;

                    for (index_t k = 0; k < numElementIntegrationPoints; ++k) {
                        A_K[i][j] += w[k]
                                     * ((DPhiInvT * grad_b_q[k][j]) * (DPhiInvT * grad_b_q[k][i]))
                                     * absDetDPhi;
                    }
                }
            }

            // 2. Compute the contribution to resultAx = A*x with A_K
            for (index_t i = 0; i < numElementDOFs; ++i) {
                index_t I = global_dofs[i];
                for (index_t j = 0; j < numElementDOFs; ++j) {
                    index_t J = global_dofs[j];

                    resultAx[i] += A_K[i][j] * x[j];
                }
            }
        }
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    void LagrangeSpace<T, Dim, Order, QuadratureType>::evaluateLoadVector(
        Vector<T, numGlobalDOFs>& b) const {
        assert(b.dim > 0);  // TODO change assert to be correct
        // TODO implement
    }

    ///////////////////////////////////////////////////////////////////////
    /// Degree of Freedom operations //////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    LagrangeSpace<T, Dim, Order, QuadratureType>::point_t
    LagrangeSpace<T, Dim, Order, QuadratureType>::getCoordsOfDOF(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& dof_index) const {
        return this->mesh_m.getVertexPosition(makeNDIndex(dof_index));
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    LagrangeSpace<T, Dim, Order, QuadratureType>::index_t
    LagrangeSpace<T, Dim, Order, QuadratureType>::getLocalDOFIndex(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& global_dof_index,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& element_index) const {
        // TODO implement
        return 0;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    LagrangeSpace<T, Dim, Order, QuadratureType>::index_t
    LagrangeSpace<T, Dim, Order, QuadratureType>::getGlobalDOFIndex(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& local_dof_index,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& element_index) const {
        // TODO implement
        return 0;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    Vector<typename LagrangeSpace<T, Dim, Order, QuadratureType>::index_t,
           LagrangeSpace<T, Dim, Order, QuadratureType>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, QuadratureType>::getLocalDOFIndices(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& element_index) const {
        // TODO implement
        return Vector<index_t, numElementDOFs>();
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    Vector<typename LagrangeSpace<T, Dim, Order, QuadratureType>::index_t,
           LagrangeSpace<T, Dim, Order, QuadratureType>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, QuadratureType>::getGlobalDOFIndices(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& element_index) const {
        // TODO implement
        return Vector<index_t, numElementDOFs>();
    }

    ///////////////////////////////////////////////////////////////////////
    /// Basis functions and gradients /////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    T LagrangeSpace<T, Dim, Order, QuadratureType>::evaluateRefElementBasis(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& localDOF,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::point_t& localPoint) const {
        // Assert that the local vertex index is valid.
        assert(localDOF < numElementVertices
               && "The local vertex index is invalid");  // TODO assumes 1st order Lagrange

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the local vertex indices for the local vertex index.
        const mesh_element_vertex_vec_t local_vertex_indices =
            this->ref_element_m.getLocalVertices()[localDOF];

        // The variable that accumulates the product of the shape functions.
        T product = 1;

        for (std::size_t d = 0; d < Dim; d++) {
            if (localPoint[d] < local_vertex_indices[d]) {
                product *= localPoint[d];
            } else {
                product *= 1.0 - localPoint[d];
            }
        }

        return product;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    LagrangeSpace<T, Dim, Order, QuadratureType>::gradient_vec_t
    LagrangeSpace<T, Dim, Order, QuadratureType>::evaluateRefElementBasisGradient(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& localDOF,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::point_t& localPoint) const {
        // TODO assumes 1st order Lagrange

        // Assert that the local vertex index is valid.
        assert(localDOF < numElementVertices && "The local vertex index is invalid");

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the local vertex indices for the local vertex index.
        const mesh_element_vertex_vec_t local_vertex_indices =
            this->ref_element_m.getLocalVertices()[localDOF];

        gradient_vec_t gradient(1);

        // To construct the gradient we need to loop over the dimensions and multiply the
        // shape functions in each dimension except the current one. The one of the current
        // dimension is replaced by the derivative of the shape function in that dimension,
        // which is either 1 or -1.
        for (std::size_t d = 0; d < Dim; d++) {
            // The variable that accumulates the product of the shape functions.
            T product = 1;

            for (std::size_t d2 = 0; d2 < Dim; d2++) {
                if (d2 == d) {
                    if (localPoint[d] < local_vertex_indices[d]) {
                        product *= 1;
                    } else {
                        product *= -1;
                    }
                } else {
                    if (localPoint[d2] < local_vertex_indices[d2]) {
                        product *= localPoint[d2];
                    } else {
                        product *= 1.0 - localPoint[d2];
                    }
                }
            }

            gradient[d] = product;
        }

        return gradient;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    NDIndex<Dim> LagrangeSpace<T, Dim, Order, QuadratureType>::makeNDIndex(
        const Vector<T, Dim>& indices) const {
        // Not sure if this is the best way, but the getVertexPosition function expects an
        // NDIndex, with the vertex index used being the first in the NDIndex. No other index is
        // used, so we can just set the first and the last to the index we actually want.
        NDIndex<Dim> nd_index;
        for (unsigned d = 0; d < Dim; ++d) {
            nd_index[d] = Index(indices[d], indices[d]);
        }
        return nd_index;
    }

}  // namespace ippl
