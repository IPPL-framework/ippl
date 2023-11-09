
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

    ///////////////////////////////////////////////////////////////////////
    /// Assembly operations ///////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    void LagrangeSpace<T, Dim, Order, QuadratureType>::evaluateAx(
        const Vector<T, numGlobalDOFs>& x, Vector<T, numGlobalDOFs>& resultAx) const {
        // List of quadrature weights
        Vector<T, QuadratureType::numElementNodes> w = this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        nd_index_t zeroNdIndex = Vector<index_t, Dim>(0);

        // Inverse Transpose Transformation Jacobian
        Vector<T, Dim> DPhiInvT = this->ref_element_m.getInverseTransposeTransformationJacobian(
            this->getElementMeshVertexIndices(zeroNdIndex));

        // Absolute value of det Phi_K
        T absDetDPhi = std::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexIndices(zeroNdIndex)));

        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<gradient_vec_t, this->numElementDOFs>, QuadratureType::numElementNodes>
            grad_b_q;
        for (index_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (index_t i = 0; i < this->numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementBasisGradient(i, q[k]);
            }
        }

        // Allocate memory for the element matrix
        Vector<Vector<T, this->numElementDOFs>, this->numElementDOFs> A_K;

        for (index_t element_index = 0; element_index < this->numElements(); ++element_index) {
            Vector<index_t, this->numElementDOFs> global_dofs;
            Vector<index_t, this->numElementDOFs> local_dofs;

            // 1. Compute the Galerkin element matrix A_K
            for (index_t i = 0; i < this->numElementDOFs; ++i) {
                for (index_t j = 0; j < this->numElementDOFs; ++j) {
                    A_K[i][j] = 0.0;

                    for (index_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                        A_K[i][j] +=
                            w[k]
                            * dot((DPhiInvT * grad_b_q[k][j]), (DPhiInvT * grad_b_q[k][i])).apply()
                            * absDetDPhi;
                    }
                }
            }

            // 2. Compute the contribution to resultAx = A*x with A_K
            for (index_t i = 0; i < this->numElementDOFs; ++i) {
                index_t I = global_dofs[i];
                for (index_t j = 0; j < this->numElementDOFs; ++j) {
                    index_t J = global_dofs[j];

                    resultAx[I] += A_K[i][j] * x[J];
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

    // template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    // LagrangeSpace<T, Dim, Order, QuadratureType>::point_t
    // LagrangeSpace<T, Dim, Order, QuadratureType>::getCoordsOfDOF(
    //     const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& dof_index) const {
    //     // TODO fix, this just did it for the vertex, not the DOF itself
    //     return this->mesh_m.getVertexPosition(makeNDIndex(dof_index));
    // }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    LagrangeSpace<T, Dim, Order, QuadratureType>::index_t
    LagrangeSpace<T, Dim, Order, QuadratureType>::getLocalDOFIndex(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& global_dof_index,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& element_index) const {
        // TODO implement
        return global_dof_index + element_index;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    LagrangeSpace<T, Dim, Order, QuadratureType>::index_t
    LagrangeSpace<T, Dim, Order, QuadratureType>::getGlobalDOFIndex(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& local_dof_index,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& element_index) const {
        // TODO implement
        return local_dof_index + element_index;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    Vector<typename LagrangeSpace<T, Dim, Order, QuadratureType>::index_t,
           LagrangeSpace<T, Dim, Order, QuadratureType>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, QuadratureType>::getLocalDOFIndices() const {
        Vector<index_t, numElementDOFs> localDOFs;

        for (std::size_t dof = 0; dof < numElementDOFs; ++dof) {
            localDOFs[dof] = dof;
        }

        return localDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    Vector<typename LagrangeSpace<T, Dim, Order, QuadratureType>::index_t,
           LagrangeSpace<T, Dim, Order, QuadratureType>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, QuadratureType>::getGlobalDOFIndices(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& elementIndex) const {
        Vector<index_t, this->numElementDOFs> globalDOFs(0);

        // get element pos
        nd_index_t elementPos = this->getElementNDIndex(elementIndex);

        // get smallest global DOF (lower left corner in 2D)
        index_t smallestGlobalDOF = elementPos[0] * Order;

        if (Dim >= 2) {
            smallestGlobalDOF += elementPos[1] * this->mesh_m.getGridsize(1) * Order;
        }

        if (Dim >= 3) {
            smallestGlobalDOF +=
                elementPos[2] * (this->mesh_m.getGridsize(1) * this->mesh_m.getGridsize(2)) * Order;
        }

        // Add the vertex DOFs
        globalDOFs[0] = smallestGlobalDOF;
        globalDOFs[1] = smallestGlobalDOF + Order;

        if (Dim >= 2) {
            globalDOFs[2] = globalDOFs[1] + this->mesh_m.getGridsize(1) * Order;
            globalDOFs[3] = globalDOFs[0] + this->mesh_m.getGridsize(1) * Order;
        }
        if (Dim >= 3) {
            globalDOFs[4] =
                globalDOFs[0] + this->mesh_m.getGridsize(1) * this->mesh_m.getGridsize(2) * Order;
            globalDOFs[5] =
                globalDOFs[1] + this->mesh_m.getGridsize(1) * this->mesh_m.getGridsize(2) * Order;
            globalDOFs[6] =
                globalDOFs[2] + this->mesh_m.getGridsize(1) * this->mesh_m.getGridsize(2) * Order;
            globalDOFs[7] =
                globalDOFs[3] + this->mesh_m.getGridsize(1) * this->mesh_m.getGridsize(2) * Order;
        }

        if (Order > 1) {
            // If the order is greater than 1, there are edge and face DOFs, otherwise the work is
            // done

            // Add the edge DOFs
            if (Dim >= 2) {
                for (std::size_t i = 0; i < Order - 1; ++i) {
                    globalDOFs[8 + i] = globalDOFs[0] + i + 1;
                    globalDOFs[8 + Order - 1 + i] =
                        globalDOFs[1] + (i + 1) * this->mesh_m.getGridsize(1);
                    globalDOFs[8 + 2 * (Order - 1) + i] = globalDOFs[2] - (i + 1);
                    globalDOFs[8 + 3 * (Order - 1) + i] =
                        globalDOFs[3] - (i + 1) * this->mesh_m.getGridsize(1);
                }
            }
            if (Dim >= 3) {
                // TODO
            }

            // Add the face DOFs
            if (Dim >= 2) {
                for (std::size_t i = 0; i < Order - 1; ++i) {
                    for (std::size_t j = 0; j < Order - 1; ++j) {
                        // TODO CHECK
                        globalDOFs[8 + 4 * (Order - 1) + i * (Order - 1) + j] =
                            globalDOFs[0] + (i + 1) + (j + 1) * this->mesh_m.getGridsize(1);
                        globalDOFs[8 + 4 * (Order - 1) + (Order - 1) * (Order - 1) + i * (Order - 1)
                                   + j] =
                            globalDOFs[1] + (i + 1) + (j + 1) * this->mesh_m.getGridsize(1);
                        globalDOFs[8 + 4 * (Order - 1) + 2 * (Order - 1) * (Order - 1)
                                   + i * (Order - 1) + j] =
                            globalDOFs[2] - (i + 1) + (j + 1) * this->mesh_m.getGridsize(1);
                        globalDOFs[8 + 4 * (Order - 1) + 3 * (Order - 1) * (Order - 1)
                                   + i * (Order - 1) + j] =
                            globalDOFs[3] - (i + 1) + (j + 1) * this->mesh_m.getGridsize(1);
                    }
                }
            }
        }

        return globalDOFs;
    }

    ///////////////////////////////////////////////////////////////////////
    /// Basis functions and gradients /////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    T LagrangeSpace<T, Dim, Order, QuadratureType>::evaluateRefElementBasis(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& localDOF,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::point_t& localPoint) const {
        // Assert that the local vertex index is valid.
        assert(localDOF < this->numElementDOFs
               && "The local vertex index is invalid");  // TODO assumes 1st order Lagrange

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the local vertex indices for the local vertex index.
        // TODO fix not order independent, only works for order 1
        // const mesh_element_vertex_vec_t local_vertex_indices =
        //    this->ref_element_m.getLocalVertices()[localDOF];

        // The variable that accumulates the product of the shape functions.
        T product = 1;

        // for (std::size_t d = 0; d < Dim; d++) {
        //    if (localPoint[d] < local_vertex_indices[d]) {
        //        product *= localPoint[d];
        //    } else {
        //        product *= 1.0 - localPoint[d];
        //    }
        //}

        return product;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    LagrangeSpace<T, Dim, Order, QuadratureType>::gradient_vec_t
    LagrangeSpace<T, Dim, Order, QuadratureType>::evaluateRefElementBasisGradient(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& localDOF,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::point_t& localPoint) const {
        // TODO assumes 1st order Lagrange

        // Assert that the local vertex index is valid.
        assert(localDOF < this->numElementDOFs && "The local vertex index is invalid");

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the local dof nd_index
        // TODO fix not order independent, only works for order 1
        // const mesh_element_vertex_vec_t local_vertex_points =
        //    this->ref_element_m.getLocalVertices();

        // const point_t& local_vertex_point = local_vertex_points[localDOF];

        gradient_vec_t gradient(1);

        // To construct the gradient we need to loop over the dimensions and multiply the
        // shape functions in each dimension except the current one. The one of the current
        // dimension is replaced by the derivative of the shape function in that dimension,
        // which is either 1 or -1.
        // for (std::size_t d = 0; d < Dim; d++) {
        //     // The variable that accumulates the product of the shape functions.
        //     T product = 1;

        //     for (std::size_t d2 = 0; d2 < Dim; d2++) {
        //         if (d2 == d) {
        //             if (localPoint[d] < local_vertex_point[d]) {
        //                 product *= 1;
        //             } else {
        //                 product *= -1;
        //             }
        //         } else {
        //             if (localPoint[d2] < local_vertex_point[d2]) {
        //                 product *= localPoint[d2];
        //             } else {
        //                 product *= 1.0 - localPoint[d2];
        //             }
        //         }
        //     }

        //     gradient[d] = product;
        // }

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
