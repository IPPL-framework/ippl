
namespace ippl {
    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    LagrangeSpace<T, Dim, Order, QuadratureType>::LagrangeSpace(
        const Mesh<T, Dim>& mesh,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::ElementType& ref_element,
        const QuadratureType& quadrature)
        : FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order), QuadratureType>(
            mesh, ref_element, quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");
    }

    ///////////////////////////////////////////////////////////////////////
    /// Degree of Freedom operations //////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    std::size_t LagrangeSpace<T, Dim, Order, QuadratureType>::numGlobalDOFs() const {
        const Vector<index_t, Dim> meshSize = this->mesh_m.getGridsize();
        std::size_t num_global_dofs         = 1;
        for (std::size_t d = 0; d < Dim; ++d) {
            num_global_dofs *= meshSize[d] * Order;
        }

        return num_global_dofs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    LagrangeSpace<T, Dim, Order, QuadratureType>::index_t
    LagrangeSpace<T, Dim, Order, QuadratureType>::getLocalDOFIndex(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& elementIndex,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& globalDOFIndex) const {
        static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");
        // TODO fix not order independent, only works for order 1
        static_assert(Order == 1, "Only order 1 is supported at the moment");

        // Get all the global DOFs for the element
        const Vector<index_t, this->numElementDOFs> global_dofs =
            this->getGlobalDOFIndices(elementIndex);

        ippl::Vector<index_t, this->numElementDOFs> dof_mapping;
        if (Dim == 1) {
            dof_mapping = {0, 1};
        } else if (Dim == 2) {
            dof_mapping = {0, 1, 3, 2};
        } else if (Dim == 3) {
            dof_mapping = {0, 1, 3, 2, 4, 5, 7, 6};
        } else {
            // throw exception
            throw std::runtime_error("FEM Lagrange Space: Dimension not supported: "
                                     + std::to_string(Dim));
        }

        // Find the global DOF in the vector and return the local DOF index
        // TODO this can be done faster since the global DOFs are sorted
        for (index_t i = 0; i < dof_mapping.dim; ++i) {
            if (global_dofs[dof_mapping[i]] == globalDOFIndex) {
                return dof_mapping[i];
            }
        }
        throw std::runtime_error("FEM Lagrange Space: Global DOF not found in specified element");
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    LagrangeSpace<T, Dim, Order, QuadratureType>::index_t
    LagrangeSpace<T, Dim, Order, QuadratureType>::getGlobalDOFIndex(
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& elementIndex,
        const LagrangeSpace<T, Dim, Order, QuadratureType>::index_t& localDOFIndex) const {
        const auto global_dofs = this->getGlobalDOFIndices(elementIndex);

        return global_dofs[localDOFIndex];
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    Vector<typename LagrangeSpace<T, Dim, Order, QuadratureType>::index_t,
           LagrangeSpace<T, Dim, Order, QuadratureType>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, QuadratureType>::getLocalDOFIndices() const {
        Vector<index_t, numElementDOFs> localDOFs;

        for (index_t dof = 0; dof < numElementDOFs; ++dof) {
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
        ndindex_t elementPos = this->getElementNDIndex(elementIndex);

        // Compute the vector to multiply the ndindex with
        ippl::Vector<std::size_t, Dim> vec(1);
        for (std::size_t d = 1; d < dim; ++d) {
            for (std::size_t d2 = d; d2 < Dim; ++d2) {
                vec[d2] *= this->mesh_m.getGridsize(d - 1);
            }
        }
        vec *= Order;  // Multiply each dimension by the order
        index_t smallestGlobalDOF = elementPos.dot(vec);

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
                for (index_t i = 0; i < Order - 1; ++i) {
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
                for (index_t i = 0; i < Order - 1; ++i) {
                    for (index_t j = 0; j < Order - 1; ++j) {
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
        static_assert(Order == 1, "Only order 1 is supported at the moment");
        // Assert that the local vertex index is valid.
        assert(localDOF < this->numElementDOFs
               && "The local vertex index is invalid");  // TODO assumes 1st order Lagrange

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the local vertex indices for the local vertex index.
        // TODO fix not order independent, only works for order 1
        const point_t ref_element_point = this->ref_element_m.getLocalVertices()[localDOF];

        // The variable that accumulates the product of the shape functions.
        T product = 1;

        for (index_t d = 0; d < Dim; d++) {
            if (localPoint[d] < ref_element_point[d]) {
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
        // TODO fix not order independent, only works for order 1
        static_assert(Order == 1 && "Only order 1 is supported at the moment");

        // Assert that the local vertex index is valid.
        assert(localDOF < this->numElementDOFs && "The local vertex index is invalid");

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the local dof nd_index
        const mesh_element_vertex_point_vec_t local_vertex_points =
            this->ref_element_m.getLocalVertices();

        const point_t& local_vertex_point = local_vertex_points[localDOF];

        gradient_vec_t gradient(1);

        // To construct the gradient we need to loop over the dimensions and multiply the
        // shape functions in each dimension except the current one. The one of the current
        // dimension is replaced by the derivative of the shape function in that dimension,
        // which is either 1 or -1.
        for (index_t d = 0; d < Dim; d++) {
            // The variable that accumulates the product of the shape functions.
            T product = 1;

            for (index_t d2 = 0; d2 < Dim; d2++) {
                if (d2 == d) {
                    if (localPoint[d] < local_vertex_point[d]) {
                        product *= 1;
                    } else {
                        product *= -1;
                    }
                } else {
                    if (localPoint[d2] < local_vertex_point[d2]) {
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
        const Vector<T, Dim>& indices) {
        // Not sure if this is the best way, but the getVertexPosition function expects an
        // NDIndex, with the vertex index used being the first in the NDIndex. No other index is
        // used, so we can just set the first and the last to the index we actually want.
        NDIndex<Dim> nd_index;
        for (unsigned d = 0; d < Dim; ++d) {
            nd_index[d] = Index(indices[d], indices[d]);
        }
        return nd_index;
    }

    ///////////////////////////////////////////////////////////////////////
    /// Assembly operations ///////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    Kokkos::View<T*> LagrangeSpace<T, Dim, Order, QuadratureType>::evaluateAx(
        Kokkos::View<const T*> x,
        const std::function<
            T(const index_t&, const index_t&,
              const Vector<Vector<T, Dim>,
                           LagrangeSpace<T, Dim, Order, QuadratureType>::NumGlobalDOFs>&)>&
            evalFunction) const {
        Kokkos::View<T*> resultAx("resultAx", this->numGlobalDOFs());

        // Allocate memory for the element matrix
        Vector<Vector<T, this->numElementDOFs>, this->numElementDOFs> A_K;

        // local DOF indices
        index_t i, j;

        // global DOF indices
        index_t I, J;

        // quadrature index
        index_t k;

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        Vector<index_t, this->numElementDOFs> global_dofs;
        Vector<index_t, this->numElementDOFs> local_dofs;

        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<gradient_vec_t, this->numElementDOFs>, QuadratureType::numElementNodes>
            grad_b_q;
        for (k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (i = 0; i < this->numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementBasisGradient(i, q[k]);
            }
        }

        const std::size_t numElements = this->numElements();
        for (index_t elementIndex = 0; elementIndex < numElements; ++elementIndex) {
            global_dofs = this->getGlobalDOFIndices(elementIndex);
            local_dofs  = this->getLocalDOFIndices();

            // 1. Compute the Galerkin element matrix A_K
            for (i = 0; i < this->numElementDOFs; ++i) {
                for (j = 0; j < this->numElementDOFs; ++j) {
                    A_K[i][j] = 0.0;
                    for (k = 0; k < QuadratureType::numElementNodes; ++k) {
                        A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                    }
                }
            }

            // DEBUG // TODO REMOVE
            // Print the Element matrix
            std::cout << "A_K = " << std::endl;
            for (i = 0; i < this->numElementDOFs; ++i) {
                for (j = 0; j < this->numElementDOFs; ++j) {
                    std::cout << A_K[i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;

            // 2. Compute the contribution to resultAx = A*x with A_K
            for (i = 0; i < this->numElementDOFs; ++i) {
                I = global_dofs[i];
                for (j = 0; j < this->numElementDOFs; ++j) {
                    J = global_dofs[j];
                    resultAx(I) += A_K[i][j] * x(J);
                }
            }
        }

        return resultAx;
    }

    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType>
    Kokkos::View<T*> LagrangeSpace<T, Dim, Order, QuadratureType>::evaluateLoadVector() const {
        Kokkos::View<T*> b("b", this->numGlobalDOFs());

        const std::size_t numElements = this->numElements();

        index_t k, i, I;

        Vector<T, this->numElementDOFs> b_K;

        Vector<index_t, this->numElementDOFs> global_dofs;
        Vector<index_t, this->numElementDOFs> local_dofs;

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        const ndindex_t zeroNdIndex = Vector<index_t, Dim>(0);

        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, this->numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (i = 0; i < this->numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementBasis(i, q[k]);
            }
        }

        // Inverse Transpose Transformation Jacobian
        const Vector<T, Dim> DPhiInvT =
            this->ref_element_m.getInverseTransposeTransformationJacobian(
                this->getElementMeshVertexIndices(zeroNdIndex));

        // Absolute value of det Phi_K
        const T absDetDPhi = std::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexIndices(zeroNdIndex)));

        // TODO move outside of LagrangeSpace class and make compatible with kokkos
        const auto f = [](const point_t& x) {
            return x[0];  // TODO change
        };                // TODO fix
        const auto eval = [absDetDPhi, f](const index_t& i, const point_t& q_k,
                                          const Vector<T, this->numElementDOFs>& basis_q_k) {
            return f(q_k) * basis_q_k[i] * absDetDPhi;
        };

        for (index_t elementIndex = 0; elementIndex < numElements; ++elementIndex) {
            global_dofs = this->getGlobalDOFIndices(elementIndex);
            local_dofs  = this->getLocalDOFIndices();

            // 1. Compute b_K
            for (i = 0; i < this->numElementDOFs; ++i) {
                b_K[i] = 0.0;
                for (k = 0; k < QuadratureType::numElementNodes; ++k) {
                    b_K[i] += w[k] * eval(i, q[k], basis_q[k]);
                }
            }

            // 2. Compute the contribution to b
            for (i = 0; i < this->numElementDOFs; ++i) {
                I = global_dofs[i];
                b(I) += b_K[i];
            }
        }

        return b;
    }

}  // namespace ippl
