
namespace ippl {

    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor,
    // and decomposes the elements among ranks according to layout.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::LagrangeSpaceFEMVector(
        UniformCartesian<T, Dim>& mesh, ElementType& ref_element, const QuadratureType& quadrature,
        const Layout_t& layout)
        : FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order), ElementType,
                             QuadratureType, FieldLHS, FieldLHS>(mesh, ref_element, quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");

        // Initialize the elementIndices view
        initializeElementIndices(layout);
    }

    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::LagrangeSpaceFEMVector(
        UniformCartesian<T, Dim>& mesh, ElementType& ref_element, const QuadratureType& quadrature)
        : FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order), ElementType,
                             QuadratureType, FieldLHS, FieldLHS>(mesh, ref_element, quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");
    }

    // LagrangeSpace initializer, to be made available to the FEMPoissonSolver 
    // such that we can call it from setRhs.
    // Sets the correct mesh ad decomposes the elements among ranks according to layout.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::initialize(
        UniformCartesian<T, Dim>& mesh, const Layout_t& layout)
    {
        FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order), ElementType,
                           QuadratureType, FieldLHS, FieldLHS>::setMesh(mesh);

        // Initialize the elementIndices view
        initializeElementIndices(layout);
    }

    // Initialize element indices Kokkos View
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                       FieldRHS>::initializeElementIndices(const Layout_t& layout) {
        layout_m = layout;
        const auto& ldom = layout.getLocalNDIndex();
        int npoints      = ldom.size();
        auto first       = ldom.first();
        auto last        = ldom.last();
        ippl::Vector<double, Dim> bounds;

        for (size_t d = 0; d < Dim; ++d) {
            bounds[d] = this->nr_m[d] - 1;
        }

        int upperBoundaryPoints = -1;

        Kokkos::View<size_t*> points("ComputeMapping", npoints);
        Kokkos::parallel_reduce(
            "ComputePoints", npoints,
            KOKKOS_CLASS_LAMBDA(const int i, int& local) {
                int idx = i;
                indices_t val;
                bool isBoundary = false;
                for (unsigned int d = 0; d < Dim; ++d) {
                    int range = last[d] - first[d] + 1;
                    val[d]    = first[d] + (idx % range);
                    idx /= range;
                    if (val[d] == bounds[d]) {
                        isBoundary = true;
                    }
                }
                points(i) = (!isBoundary) * (this->getElementIndex(val));
                local += isBoundary;
            },
            Kokkos::Sum<int>(upperBoundaryPoints));
        Kokkos::fence();

        int elementsPerRank = npoints - upperBoundaryPoints;
        elementIndices      = Kokkos::View<size_t*>("i", elementsPerRank);
        Kokkos::View<size_t> index("index");

        Kokkos::parallel_for(
            "RemoveNaNs", npoints, KOKKOS_CLASS_LAMBDA(const int i) {
                if ((points(i) != 0) || (i == 0)) {
                    const size_t idx    = Kokkos::atomic_fetch_add(&index(), 1);
                    elementIndices(idx) = points(i);
                }
            });
    }

    ///////////////////////////////////////////////////////////////////////
    /// Degree of Freedom operations //////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                         FieldRHS>::numGlobalDOFs() const {
        size_t num_global_dofs = 1;
        for (size_t d = 0; d < Dim; ++d) {
            num_global_dofs *= this->nr_m[d] * Order;
        }

        return num_global_dofs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    size_t LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::getLocalDOFIndex
    (const size_t& elementIndex, const size_t& globalDOFIndex) const {
        static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");
        // TODO fix not order independent, only works for order 1
        static_assert(Order == 1, "Only order 1 is supported at the moment");

        // Get all the global DOFs for the element
        const Vector<size_t, this->numElementDOFs> global_dofs =
            this->getGlobalDOFIndices(elementIndex);

        ippl::Vector<size_t, this->numElementDOFs> dof_mapping;
        if (Dim == 1) {
            dof_mapping = {0, 1};
        } else if (Dim == 2) {
            dof_mapping = {0, 1, 3, 2};
        } else if (Dim == 3) {
            dof_mapping = {0, 1, 3, 2, 4, 5, 7, 6};
        }

        // Find the global DOF in the vector and return the local DOF index
        // TODO this can be done faster since the global DOFs are sorted
        for (size_t i = 0; i < dof_mapping.dim; ++i) {
            if (global_dofs[dof_mapping[i]] == globalDOFIndex) {
                return dof_mapping[i];
            }
        }
        return std::numeric_limits<size_t>::quiet_NaN();
        //throw IpplException("LagrangeSpaceFEMVector::getLocalDOFIndex()",
        //                    "FEM Lagrange Space: Global DOF not found in specified element");
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t
    LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::getGlobalDOFIndex(const size_t& elementIndex,
                                               const size_t& localDOFIndex) const {
        const auto global_dofs = this->getGlobalDOFIndices(elementIndex);

        return global_dofs[localDOFIndex];
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION Vector<size_t, LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType,
                                                 FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::getLocalDOFIndices() const {
        Vector<size_t, numElementDOFs> localDOFs;

        for (size_t dof = 0; dof < numElementDOFs; ++dof) {
            localDOFs[dof] = dof;
        }

        return localDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION Vector<size_t, LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType,
                                                 FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::getGlobalDOFIndices(const size_t& elementIndex) const {
        Vector<size_t, this->numElementDOFs> globalDOFs(0);

        // get element pos
        indices_t elementPos = this->getElementNDIndex(elementIndex);

        // Compute the vector to multiply the ndindex with
        ippl::Vector<size_t, Dim> vec(1);
        for (size_t d = 1; d < dim; ++d) {
            for (size_t d2 = d; d2 < Dim; ++d2) {
                vec[d2] *= this->nr_m[d - 1];
            }
        }
        vec *= Order;  // Multiply each dimension by the order
        size_t smallestGlobalDOF = elementPos.dot(vec);

        // Add the vertex DOFs
        globalDOFs[0] = smallestGlobalDOF;
        globalDOFs[1] = smallestGlobalDOF + Order;

        if (Dim >= 2) {
            globalDOFs[2] = globalDOFs[1] + this->nr_m[1] * Order;
            globalDOFs[3] = globalDOFs[0] + this->nr_m[1] * Order;
        }
        if (Dim >= 3) {
            globalDOFs[4] = globalDOFs[0] + this->nr_m[1] * this->nr_m[2] * Order;
            globalDOFs[5] = globalDOFs[1] + this->nr_m[1] * this->nr_m[2] * Order;
            globalDOFs[6] = globalDOFs[2] + this->nr_m[1] * this->nr_m[2] * Order;
            globalDOFs[7] = globalDOFs[3] + this->nr_m[1] * this->nr_m[2] * Order;
        }

        if (Order > 1) {
            // If the order is greater than 1, there are edge and face DOFs, otherwise the work is
            // done

            // Add the edge DOFs
            if (Dim >= 2) {
                for (size_t i = 0; i < Order - 1; ++i) {
                    globalDOFs[8 + i]                   = globalDOFs[0] + i + 1;
                    globalDOFs[8 + Order - 1 + i]       = globalDOFs[1] + (i + 1) * this->nr_m[1];
                    globalDOFs[8 + 2 * (Order - 1) + i] = globalDOFs[2] - (i + 1);
                    globalDOFs[8 + 3 * (Order - 1) + i] = globalDOFs[3] - (i + 1) * this->nr_m[1];
                }
            }
            if (Dim >= 3) {
                // TODO
            }

            // Add the face DOFs
            if (Dim >= 2) {
                for (size_t i = 0; i < Order - 1; ++i) {
                    for (size_t j = 0; j < Order - 1; ++j) {
                        // TODO CHECK
                        globalDOFs[8 + 4 * (Order - 1) + i * (Order - 1) + j] =
                            globalDOFs[0] + (i + 1) + (j + 1) * this->nr_m[1];
                        globalDOFs[8 + 4 * (Order - 1) + (Order - 1) * (Order - 1) + i * (Order - 1)
                                   + j] = globalDOFs[1] + (i + 1) + (j + 1) * this->nr_m[1];
                        globalDOFs[8 + 4 * (Order - 1) + 2 * (Order - 1) * (Order - 1)
                                   + i * (Order - 1) + j] =
                            globalDOFs[2] - (i + 1) + (j + 1) * this->nr_m[1];
                        globalDOFs[8 + 4 * (Order - 1) + 3 * (Order - 1) * (Order - 1)
                                   + i * (Order - 1) + j] =
                            globalDOFs[3] - (i + 1) + (j + 1) * this->nr_m[1];
                    }
                }
            }
        }

        return globalDOFs;
    }

    ///////////////////////////////////////////////////////////////////////
    /// Basis functions and gradients /////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION T
    LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        evaluateRefElementShapeFunction(
            const size_t& localDOF,
            const LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                FieldRHS>::point_t& localPoint) const {
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

        for (size_t d = 0; d < Dim; d++) {
            if (localPoint[d] < ref_element_point[d]) {
                product *= localPoint[d];
            } else {
                product *= 1.0 - localPoint[d];
            }
        }

        return product;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION typename LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                           FieldRHS>::point_t
    LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        evaluateRefElementShapeFunctionGradient(
            const size_t& localDOF,
            const LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                FieldRHS>::point_t& localPoint) const {
        // TODO fix not order independent, only works for order 1
        static_assert(Order == 1 && "Only order 1 is supported at the moment");

        // Assert that the local vertex index is valid.
        assert(localDOF < this->numElementDOFs && "The local vertex index is invalid");

        assert(this->ref_element_m.isPointInRefElement(localPoint)
               && "Point is not in reference element");

        // Get the local dof nd_index
        const vertex_points_t local_vertex_points = this->ref_element_m.getLocalVertices();

        const point_t& local_vertex_point = local_vertex_points[localDOF];

        point_t gradient(1);

        // To construct the gradient we need to loop over the dimensions and multiply the
        // shape functions in each dimension except the current one. The one of the current
        // dimension is replaced by the derivative of the shape function in that dimension,
        // which is either 1 or -1.
        for (size_t d = 0; d < Dim; d++) {
            // The variable that accumulates the product of the shape functions.
            T product = 1;

            for (size_t d2 = 0; d2 < Dim; d2++) {
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

    ///////////////////////////////////////////////////////////////////////
    /// Assembly operations ///////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx(FieldLHS& vector, F& evalFunction) const {
        Inform m("");
        
        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);


        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultVector = vector.deepCopy();
        resultVector = 0;

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, this->numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = vector.getView();
        AtomicViewType resultView = resultVector.getView();


        // Get domain information
        auto ldom = layout_m.getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);


        // information to translate the ndindices to FEMVector positions.
        // We add the plus one, because we have that last does not correspond
        // to the position end+1, but simply to end
        ippl::Vector<size_t, Dim> extents = (ldom.last()+1) - ldom.first() + 2;
        ippl::Vector<size_t, Dim> v(1);
        for (size_t i = 1; i < Dim; ++i) {
            for(size_t j = i; j < Dim; ++j) {
                v(j) *= extents(i-1);
            }
        }

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex                            = elementIndices(index);
                const Vector<size_t, this->numElementDOFs> local_dof = this->getLocalDOFIndices();
                const Vector<size_t, this->numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);
                Vector<indices_t, this->numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < this->numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices
                size_t i, j;

                // Element matrix
                Vector<Vector<T, this->numElementDOFs>, this->numElementDOFs> A_K;

                // 1. Compute the Galerkin element matrix A_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    for (j = 0; j < this->numElementDOFs; ++j) {
                        A_K[i][j] = 0.0;
                        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                            A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                        }
                    }
                }

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;
                size_t I, J;
                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Skip boundary DOFs (Zero Dirichlet BCs)
                    if (this->isDOFOnBoundary(I_nd)) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    I = 0;
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + 1;
                        I += v[d]*I_nd[d];
                    }

                    for (j = 0; j < this->numElementDOFs; ++j) {
                        J_nd = global_dof_ndindices[j];

                        // Skip boundary DOFs (Zero Dirichlet BCs)
                        if (this->isDOFOnBoundary(J_nd)) {
                            continue;
                        }

                        // get the appropriate index for the Kokkos view of the field
                        J = 0;
                        for (unsigned d = 0; d < Dim; ++d) {
                            J_nd[d] = J_nd[d] - ldom[d].first() + 1;
                            J += v[d]*J_nd[d];
                        }

                        resultView(I) += A_K[i][j] * view(J);
                    }
                }
            });
        IpplTimings::stopTimer(outer_loop);

        IpplTimings::stopTimer(evalAx);
        
        return resultVector;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                       FieldRHS>::evaluateLoadVector(FieldLHS& vector) const {
        Inform m("");
        
        // start a timer
        static IpplTimings::TimerRef evalLoadV = IpplTimings::getTimer("evaluateLoadVector");
        IpplTimings::startTimer(evalLoadV);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        const indices_t zeroNdIndex = Vector<size_t, Dim>(0);

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, this->numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Get domain information and ghost cells
        auto ldom        = layout_m.getLocalNDIndex();


        // Get boundary conditions from field
        FieldLHS temp_vector = vector.deepCopy();
        temp_vector = 0;

        // Get field data and make it atomic,
        // since it will be added to during the kokkos loop
        // We work with a temporary field since we need to use field
        // to evaluate the load vector; then we assign temp to RHS field
        AtomicViewType atomic_view = temp_vector.getView();
        ViewType view = vector.getView(); 

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop =
            IpplTimings::getTimer("evaluateLoadVec: outer loop");
        IpplTimings::startTimer(outer_loop);
        
        // information to translate the ndindices to FEMVector positions.
        // We add the plus one, because we have that last does not correspond
        // to the position end+1, but simply to end
        ippl::Vector<size_t, Dim> extents = (ldom.last()+1) - ldom.first() + 2;
        ippl::Vector<size_t, Dim> v(1);
        for (size_t i = 1; i < Dim; ++i) {
            for(size_t j = i; j < Dim; ++j) {
                v(j) *= extents(i-1);
            }
        }
        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(size_t index) {
                const size_t elementIndex                              = elementIndices(index);
                const Vector<size_t, this->numElementDOFs> local_dofs  = this->getLocalDOFIndices();
                const Vector<size_t, this->numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);

                size_t i, I;
                size_t II, JJ;

                // 1. Compute b_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    I = global_dofs[i];

                    // TODO fix for higher order
                    auto dof_ndindex_I = this->getMeshVertexNDIndex(I);

                    if (this->isDOFOnBoundary(dof_ndindex_I)) {
                        continue;
                    }

                    // calculate the contribution of this element
                    T contrib = 0;
                    for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                        T val = 0;
                        for (size_t j = 0; j < this->numElementDOFs; ++j) {
                            // get field index corresponding to this DOF
                            size_t J           = global_dofs[j];
                            auto dof_ndindex_J = this->getMeshVertexNDIndex(J);
                            JJ = 0;
                            
                            for (unsigned d = 0; d < Dim; ++d) {
                                dof_ndindex_J[d] = dof_ndindex_J[d] - ldom[d].first() + 1;
                                JJ += dof_ndindex_J[d]*v[d];
                            }
                            //std::cout << dof_ndindex_J << " on " << ldom.first() << "x" << ldom.last() << " -> " << JJ << "\n";

                            // get field value at DOF and interpolate to q_k
                            val += basis_q[k][j] * view(JJ);
                        }

                        contrib += w[k] * basis_q[k][i] * absDetDPhi * val;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    II = 0;
                    for (unsigned d = 0; d < Dim; ++d) {
                        dof_ndindex_I[d] = dof_ndindex_I[d] - ldom[d].first() + 1;
                        II += dof_ndindex_I[d]*v[d];
                    }
                    //std::cout << dof_ndindex_I << " on " << ldom.first() << "x" << ldom.last() << " -> " << II << "\n";

                    // add the contribution of the element to the field
                    atomic_view(II) += contrib;

                }
            });
        IpplTimings::stopTimer(outer_loop);

        vector = temp_vector;

        IpplTimings::stopTimer(evalLoadV);
        
    }


    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    FieldLHS LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
        FieldRHS>::interpolateToFEMVector(const FieldRHS& field) const {
        
        // The Kokkos view.
        auto view = field.getView();
        
        // figure out the total number of elements in the field.
        size_t n = 1;
        for (size_t d = 0; d < Dim; ++d) {
            n *= view.extent(d);
        }

        // Next we need to define how we are going to do the index mappings, for
        // this we rely on the mapping from ndIndex to number.
        ippl::Vector<size_t, Dim> v(1);
        for (size_t i = 1; i < Dim; ++i) {
            for(size_t j = i; j < Dim; ++j) {
                v(j) *= view.extent(i-1);
            }
        }
        /*
        if (ippl::Comm->rank() == 0) {
            std::cout << "extents: " << view.extent(0) << "x" << view.extent(1)  << "\n";
        }
        */
        auto& layout = field.getLayout();
        auto ldom = layout.getLocalNDIndex();
        /*
        if (ippl::Comm->rank() == 0) {
            std::cout << "domain decomposition\n";
        }
        ippl::Comm->barrier();

        for (size_t r = 0; r < ippl::Comm->size(); ++r) {
            if (r == ippl::Comm->rank()) {
                std::cout << r << ": " << ldom.first() << ", " << ldom.last() << "\n";
            }
            ippl::Comm->barrier();
        }
        */

        // Next up we need to create the neighbor thing and get all the indices
        auto neighbors = layout.getNeighbors();
        auto neighborSendRange = layout.getNeighborsSendRange();
        auto neighborRecvRange = layout.getNeighborsRecvRange();
        std::vector<size_t> neighborsFV;
        std::vector< Kokkos::View<size_t*> > sendIdxs;
        std::vector< Kokkos::View<size_t*> > recvIdxs;
        std::vector< std::vector<size_t> > sendIdxsTemp;
        std::vector< std::vector<size_t> > recvIdxsTemp;
        /*
        if (ippl::Comm->rank() == 0) {
            std::cout << "comm layout:\n";
        }
        */
        for (size_t i = 0; i < neighbors.size(); ++i) {
            const auto& componentNeighbors = neighbors[i];
            for (size_t j = 0; j < componentNeighbors.size(); ++j) {
                int rank = componentNeighbors[j];
                // check if we already have a rank in there
                const auto it = std::find(neighborsFV.begin(), neighborsFV.end(), rank);
                size_t idx = it - neighborsFV.begin();
                if (it == neighborsFV.end()) {
                    // it is not yet in
                    neighborsFV.push_back(rank);
                    sendIdxsTemp.push_back(std::vector<size_t>());
                    recvIdxsTemp.push_back(std::vector<size_t>());
                }


                typename Layout_t::bound_type sendRange = neighborSendRange[i][j];
                /*
                if (ippl::Comm->rank() == 0) {
                    std::cout << "s: 0->" << rank << ": [(" << sendRange.lo[0] << "," <<
                                sendRange.lo[1] << "), (" <<
                                sendRange.hi[0] << "," << sendRange.hi[1] << ")]\n";
                }
                */

                if (Dim == 1) {
                    for (size_t x = sendRange.lo[0]; x < sendRange.hi[0]; x++) {
                        sendIdxsTemp[idx].push_back(x);
                    }
                } else if (Dim == 2) {
                    for (size_t x = sendRange.lo[0]; x < sendRange.hi[0]; x++) {
                        for (size_t y = sendRange.lo[1]; y < sendRange.hi[1]; y++) {
                            sendIdxsTemp[idx].push_back(x*v[0] + y*v[1]);
                        }
                    }
                } else if (Dim == 3) {
                    for (size_t x = sendRange.lo[0]; x < sendRange.hi[0]; x++) {
                        for (size_t y = sendRange.lo[1]; y < sendRange.hi[1]; y++) {
                            for (size_t z = sendRange.lo[2]; z < sendRange.hi[2]; z++) {
                                sendIdxsTemp[idx].push_back(x*v[0] + y*v[1] + z*v[2]);
                            }
                        }
                    }
                }


                typename Layout_t::bound_type recvRange = neighborRecvRange[i][j];
                /*
                if (ippl::Comm->rank() == 0) {
                    std::cout << "r: " << rank <<"->0" << ": [(" << recvRange.lo[0] << "," <<
                                recvRange.lo[1] << "), (" <<
                                recvRange.hi[0] << "," << recvRange.hi[1] << ")]\n";
                }
                */

                if (Dim == 1) {
                    for (size_t x = recvRange.lo[0]; x < recvRange.hi[0]; x++) {
                        recvIdxsTemp[idx].push_back(x);
                    }
                } else if (Dim == 2) {
                    for (size_t x = recvRange.lo[0]; x < recvRange.hi[0]; x++) {
                        for (size_t y = recvRange.lo[1]; y < recvRange.hi[1]; y++) {
                            recvIdxsTemp[idx].push_back(x*v[0] + y*v[1]);
                            /*
                            if (ippl::Comm->rank() == 0) {
                                std::cout << "adding: " << x*v[0] + y*v[1] << " to vec of size" <<
                                            n << "\n";
                            }
                            */
                        }
                    }
                } else if (Dim == 3) {
                    for (size_t x = recvRange.lo[0]; x < recvRange.hi[0]; x++) {
                        for (size_t y = recvRange.lo[1]; y < recvRange.hi[1]; y++) {
                            for (size_t z = recvRange.lo[2]; z < recvRange.hi[2]; z++) {
                                recvIdxsTemp[idx].push_back(x*v[0] + y*v[1] + z*v[2]);
                            }
                        }
                    }
                }
                

                
            }
        }

        // now copy all the data from the Temps to the real deal
        for (size_t i = 0; i < neighborsFV.size(); ++i) {
            sendIdxs.push_back(Kokkos::View<size_t*>("FEMvector::sendIdxs[" + std::to_string(i) +
                                                     "]", sendIdxsTemp[i].size()));
            recvIdxs.push_back(Kokkos::View<size_t*>("FEMvector::recvIdxs[" + std::to_string(i) +
                                                     "]", recvIdxsTemp[i].size()));
            auto sendView = sendIdxs[i];
            auto recvView = recvIdxs[i];
            auto hSendView = Kokkos::create_mirror_view(sendView);
            auto hRecvView = Kokkos::create_mirror_view(recvView);

            for (size_t j = 0; j < sendIdxsTemp[i].size(); ++j) {
                hSendView(j) = sendIdxsTemp[i][j];
            }

            for (size_t j = 0; j < recvIdxsTemp[i].size(); ++j) {
                hRecvView(j) = recvIdxsTemp[i][j];
            }

            Kokkos::deep_copy(sendView, hSendView);
            Kokkos::deep_copy(recvView, hRecvView);
        }

        
        // Now finaly create the FEMVector
        FEMVector<T> vec(n, neighborsFV, sendIdxs, recvIdxs);

        // set the values of the FEMVector
        auto vecView = vec.getView();
        auto fieldView = field.getView();
        using index_array_type =
                typename RangePolicy<Dim, typename FieldRHS::execution_space>::index_array_type;
        ippl::parallel_for(
            "LagrangeSpaceFEMVector::interpolateToFEMVector", getRangePolicy(fieldView),
            KOKKOS_LAMBDA(const index_array_type& args) {
                size_t idx = 0;
                for (unsigned i = 0; i < Dim; ++i) {
                    idx += args[i]*v[i];
                }
                vecView[idx] = apply(fieldView, args);
            }
        );

        return vec;

    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
        FieldRHS>::reconstructToField(const FEMVector<T>& vector, FieldRHS& field) const {
        

        // Next we need to define how we are going to do the index mappings, for
        // this we rely on the mapping from ndIndex to number.
        auto fieldView = field.getView();

        ippl::Vector<size_t, Dim> v(1);
        for (size_t i = 1; i < Dim; ++i) {
            for(size_t j = i; j < Dim; ++j) {
                v(j) *= fieldView.extent(i-1);
            }
        }

        
        // Copy data over
        auto vecView = vector.getView();
        using index_array_type =
                typename RangePolicy<Dim, typename FieldRHS::execution_space>::index_array_type;
        ippl::parallel_for(
            "LagrangeSpaceFEMVector::reconstructToFEMVector", getRangePolicy(fieldView),
            KOKKOS_LAMBDA(const index_array_type& args) {
                size_t idx = 0;
                for (unsigned i = 0; i < Dim; ++i) {
                    idx += args[i]*v[i];
                }

                apply(fieldView, args) = vecView[idx];
            }
        );        
    }


/*
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    std::function<T(size_t,size_t,size_t)> LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::diffusionOperator() const {
        
        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();
        
        Vector<Vector<point_t, this->numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        const Vector<size_t, Dim> zeroNdIndex = Vector<size_t, Dim>(0);

        // We can pass the zeroNdIndex here, since the transformation jacobian does not depend
        // on translation
        const auto firstElementVertexPoints =
            this->getElementMeshVertexPoints(zeroNdIndex);

        // Compute Inverse Transpose Transformation Jacobian ()
        const Vector<T, Dim> DPhiInvT =
            this->ref_element_m.getInverseTransposeTransformationJacobian(firstElementVertexPoints);

        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        const T absDetDPhi = Kokkos::abs(
            this->ref_element_m.getDeterminantOfTransformationJacobian(firstElementVertexPoints));

        auto f = KOKKOS_CLASS_LAMBDA(size_t i, size_t j, size_t q) -> T {
            return w[q]*dot(DPhiInvT*grad_b_q[q][j], DPhiInvT*grad_b_q[q][i]).apply()*absDetDPhi;
        };

        return f;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename Functor>
    std::function<T(size_t,size_t,ippl::Vector<T,Dim>)> LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::loadOperator(Functor f) const {
        
        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();


        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, this->numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        const Vector<size_t, Dim> zeroNdIndex = Vector<size_t, Dim>(0);

        // We can pass the zeroNdIndex here, since the transformation jacobian does not depend
        // on translation
        const auto firstElementVertexPoints =
            this->getElementMeshVertexPoints(zeroNdIndex);

        // Compute Inverse Transpose Transformation Jacobian ()
        const Vector<T, Dim> DPhiInvT =
            this->ref_element_m.getInverseTransposeTransformationJacobian(firstElementVertexPoints);

        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        const T absDetDPhi = Kokkos::abs(
            this->ref_element_m.getDeterminantOfTransformationJacobian(firstElementVertexPoints));

        auto loadF = KOKKOS_CLASS_LAMBDA(size_t i, size_t q, point_t x) -> T {
            return w[q]*basis_q[q][i]* f(x)*absDetDPhi;
        };

        return loadF;
    }

*/
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    T LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::computeError(
        const FieldRHS& u_h, const F& u_sol) const {
        if (this->quadrature_m.getOrder() < (2 * Order + 1)) {
            // throw exception
            throw IpplException(
                "LagrangeSpaceFEMVector::computeError()",
                "Order of quadrature rule for error computation should be > 2*p + 1");
        }

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, this->numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        const indices_t zeroNdIndex = Vector<size_t, Dim>(0);

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Variable to sum the error to
        T error = 0;

        // Get domain information and ghost cells
        auto ldom        = (u_h.getLayout()).getLocalNDIndex();
        const int nghost = u_h.getNghost();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // Loop over elements to compute contributions
        Kokkos::parallel_reduce(
            "Compute error over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(size_t index, double& local) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, this->numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);

                // contribution of this element to the error
                T contrib = 0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    T val_u_sol = u_sol(this->ref_element_m.localToGlobal(
                        this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex)),
                        q[k]));

                    T val_u_h = 0;
                    for (size_t i = 0; i < this->numElementDOFs; ++i) {
                        // get field index corresponding to this DOF
                        size_t I           = global_dofs[i];
                        auto dof_ndindex_I = this->getMeshVertexNDIndex(I);
                        for (unsigned d = 0; d < Dim; ++d) {
                            dof_ndindex_I[d] = dof_ndindex_I[d] - ldom[d].first() + nghost;
                        }

                        // get field value at DOF and interpolate to q_k
                        val_u_h += basis_q[k][i] * apply(u_h, dof_ndindex_I);
                    }

                    contrib += w[k] * Kokkos::pow(val_u_sol - val_u_h, 2) * absDetDPhi;
                }
                local += contrib;
            },
            Kokkos::Sum<double>(error));

        // MPI reduce
        T global_error = 0.0;
        Comm->allreduce(error, global_error, 1, std::plus<T>());

        return Kokkos::sqrt(global_error);
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    T LagrangeSpaceFEMVector<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                    FieldRHS>::computeErrorInf(const FieldRHS& u_h, const F& u_sol) const {
        if (this->quadrature_m.getOrder() < (2 * Order + 1)) {
            // throw exception
            throw IpplException(
                "LagrangeSpaceFEMVector::computeError()",
                "Order of quadrature rule for error computation should be > 2*p + 1");
        }

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, this->numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        const indices_t zeroNdIndex = Vector<size_t, Dim>(0);

        // Variable to sum the error to
        T error = 0;

        // Get domain information and ghost cells
        auto ldom        = (u_h.getLayout()).getLocalNDIndex();
        const int nghost = u_h.getNghost();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // Loop over elements to compute contributions
        Kokkos::parallel_reduce(
            "Compute error over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(size_t index, double& local_max) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, this->numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);
                auto dof_points =
                    this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex));

                for (size_t i = 0; i < this->numElementDOFs; ++i) {
                    size_t I           = global_dofs[i];
                    auto dof_ndindex_I = this->getMeshVertexNDIndex(I);
                    for (unsigned d = 0; d < Dim; ++d) {
                        dof_ndindex_I[d] = dof_ndindex_I[d] - ldom[d].first() + nghost;
                    }

                    // computed field value at DOF
                    T val_u_h = apply(u_h, dof_ndindex_I);

                    // solution field value at DOF
                    T val_u_sol = u_sol(dof_points[i]);

                    T local_norm = Kokkos::abs(val_u_h - val_u_sol);

                    if (local_norm > local_max) {
                        local_max = local_norm;
                    }
                }
            },
            Kokkos::Max<double>(error));

        // MPI reduce
        T global_error = 0.0;
        Comm->allreduce(error, global_error, 1, std::greater<T>());

        return global_error;
    }

}  // namespace ippl
