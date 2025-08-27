
namespace ippl {

    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor,
    // and decomposes the elements among ranks according to layout.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::LagrangeSpace(
        UniformCartesian<T, Dim>& mesh, ElementType& ref_element, const QuadratureType& quadrature,
        const Layout_t& layout)
        : FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order), ElementType,
                             QuadratureType, FieldLHS, FieldRHS>(mesh, ref_element, quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");

        // Initialize the elementIndices view
        initializeElementIndices(layout);
    }

    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::LagrangeSpace(
        UniformCartesian<T, Dim>& mesh, ElementType& ref_element, const QuadratureType& quadrature)
        : FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order), ElementType,
                             QuadratureType, FieldLHS, FieldRHS>(mesh, ref_element, quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");
    }

    // LagrangeSpace initializer, to be made available to the FEMPoissonSolver 
    // such that we can call it from setRhs.
    // Sets the correct mesh ad decomposes the elements among ranks according to layout.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::initialize(
        UniformCartesian<T, Dim>& mesh, const Layout_t& layout)
    {
        FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order), ElementType,
                           QuadratureType, FieldLHS, FieldRHS>::setMesh(mesh);

        // Initialize the elementIndices view
        initializeElementIndices(layout);
    }

    // Initialize element indices Kokkos View by distributing elements among MPI ranks.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                       FieldRHS>::initializeElementIndices(const Layout_t& layout) {
        const auto& ldom = layout.getLocalNDIndex();
        int npoints      = ldom.size();
        auto first       = ldom.first();
        auto last        = ldom.last();
        ippl::Vector<double, Dim> bounds;

        for (size_t d = 0; d < Dim; ++d) {
            bounds[d] = this->nr_m[d] - 1;
        }

        int upperBoundaryPoints = -1;

        // We iterate over the local domain points, getting the corresponding elements, 
        // while tagging upper boundary points such that they can be removed after.
        Kokkos::View<size_t*> points("npoints", npoints);
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

        // The elementIndices will be the same array as computed above,
        // with the tagged upper boundary points removed.
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
    KOKKOS_FUNCTION size_t LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
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
    size_t LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::getLocalDOFIndex
    (const size_t& elementIndex, const size_t& globalDOFIndex) const {
        static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Dim must be 1, 2 or 3");
        // TODO fix not order independent, only works for order 1
        static_assert(Order == 1, "Only order 1 is supported at the moment");

        // Get all the global DOFs for the element
        const Vector<size_t, numElementDOFs> global_dofs =
            this->getGlobalDOFIndices(elementIndex);

        ippl::Vector<size_t, numElementDOFs> dof_mapping;
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
        //throw IpplException("LagrangeSpace::getLocalDOFIndex()",
        //                    "FEM Lagrange Space: Global DOF not found in specified element");
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::getGlobalDOFIndex(const size_t& elementIndex,
                                               const size_t& localDOFIndex) const {
        const auto global_dofs = this->getGlobalDOFIndices(elementIndex);

        return global_dofs[localDOFIndex];
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION Vector<size_t, LagrangeSpace<T, Dim, Order, ElementType, QuadratureType,
                                                 FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::getLocalDOFIndices() const {
        Vector<size_t, numElementDOFs> localDOFs;

        for (size_t dof = 0; dof < numElementDOFs; ++dof) {
            localDOFs[dof] = dof;
        }

        return localDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION Vector<size_t, LagrangeSpace<T, Dim, Order, ElementType, QuadratureType,
                                                 FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                  FieldRHS>::getGlobalDOFIndices(const size_t& elementIndex) const {
        Vector<size_t, numElementDOFs> globalDOFs(0);

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
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        evaluateRefElementShapeFunction(
            const size_t& localDOF,
            const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                FieldRHS>::point_t& localPoint) const {
        static_assert(Order == 1, "Only order 1 is supported at the moment");
        // Assert that the local vertex index is valid.
        assert(localDOF < numElementDOFs
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
    KOKKOS_FUNCTION typename LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                           FieldRHS>::point_t
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        evaluateRefElementShapeFunctionGradient(
            const size_t& localDOF,
            const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                FieldRHS>::point_t& localPoint) const {
        // TODO fix not order independent, only works for order 1
        static_assert(Order == 1 && "Only order 1 is supported at the moment");

        // Assert that the local vertex index is valid.
        assert(localDOF < numElementDOFs && "The local vertex index is invalid");

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
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get boundary conditions from field
        BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType = bcField[0]->getBCType();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex                            = elementIndices(index);
                const Vector<size_t, numElementDOFs> local_dof = this->getLocalDOFIndices();
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices (both i and j go from 0 to numDOFs-1 in the element)
                size_t i, j;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Handle boundary DOFs
                    // If Zero Dirichlet BCs, skip this DOF
                    // If Constant Dirichlet BCs, identity
                    if ((bcType == CONSTANT_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                        }
                        apply(resultView, I_nd) =  apply(view, I_nd);
                        continue;
                    } else if ((bcType == ZERO_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }

                    for (j = 0; j < numElementDOFs; ++j) {
                        J_nd = global_dof_ndindices[j];

                        // Skip boundary DOFs (Zero & Constant Dirichlet BCs)
                        if (((bcType == ZERO_FACE) || (bcType == CONSTANT_FACE)) 
                            && this->isDOFOnBoundary(J_nd)) {
                            continue;
                        }

                        // get the appropriate index for the Kokkos view of the field
                        for (unsigned d = 0; d < Dim; ++d) {
                            J_nd[d] = J_nd[d] - ldom[d].first() + nghost;
                        }

                        apply(resultView, I_nd) += A_K[i][j] * apply(view, J_nd);
                    }
                }
            });
        IpplTimings::stopTimer(outer_loop);

        if (bcType == PERIODIC_FACE) {
            resultField.accumulateHalo();
            bcField.apply(resultField);
            bcField.assignGhostToPhysical(resultField);
        } else {
            resultField.accumulateHalo_noghost();
        }

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_lower(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get boundary conditions from field
        BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType = bcField[0]->getBCType();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex                            = elementIndices(index);
                const Vector<size_t, numElementDOFs> local_dof = this->getLocalDOFIndices();
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices
                size_t i, j;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Handle boundary DOFs
                    // If Zero Dirichlet BCs, skip this DOF
                    // If Constant Dirichlet BCs, identity
                    if ((bcType == CONSTANT_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                        }
                        apply(resultView, I_nd) =  apply(view, I_nd);
                        continue;
                    } else if ((bcType == ZERO_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }

                    for (j = 0; j < numElementDOFs; ++j) {
                        J_nd = global_dof_ndindices[j];

                        if (global_dofs[i] >= global_dofs[j]) {
                            continue;
                        }

                        // Skip boundary DOFs (Zero & Constant Dirichlet BCs)
                        if (((bcType == ZERO_FACE) || (bcType == CONSTANT_FACE)) 
                            && this->isDOFOnBoundary(J_nd)) {
                            continue;
                        }

                        // get the appropriate index for the Kokkos view of the field
                        for (unsigned d = 0; d < Dim; ++d) {
                            J_nd[d] = J_nd[d] - ldom[d].first() + nghost;
                        }

                        apply(resultView, I_nd) += A_K[i][j] * apply(view, J_nd);
                    }
                }
            });
        IpplTimings::stopTimer(outer_loop);

        if (bcType == PERIODIC_FACE) {
            resultField.accumulateHalo();
            bcField.apply(resultField);
            bcField.assignGhostToPhysical(resultField);
        } else {
            resultField.accumulateHalo_noghost();
        }

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_upper(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get boundary conditions from field
        BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType = bcField[0]->getBCType();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex                            = elementIndices(index);
                const Vector<size_t, numElementDOFs> local_dof = this->getLocalDOFIndices();
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices
                size_t i, j;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Handle boundary DOFs
                    // If Zero Dirichlet BCs, skip this DOF
                    // If Constant Dirichlet BCs, identity
                    if ((bcType == CONSTANT_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                        }
                        apply(resultView, I_nd) =  apply(view, I_nd);
                        continue;
                    } else if ((bcType == ZERO_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }

                    for (j = 0; j < numElementDOFs; ++j) {
                        J_nd = global_dof_ndindices[j];

                        if (global_dofs[i] <= global_dofs[j]) {
                            continue;
                        }

                        // Skip boundary DOFs (Zero & Constant Dirichlet BCs)
                        if (((bcType == ZERO_FACE) || (bcType == CONSTANT_FACE)) 
                            && this->isDOFOnBoundary(J_nd)) {
                            continue;
                        }

                        // get the appropriate index for the Kokkos view of the field
                        for (unsigned d = 0; d < Dim; ++d) {
                            J_nd[d] = J_nd[d] - ldom[d].first() + nghost;
                        }

                        apply(resultView, I_nd) += A_K[i][j] * apply(view, J_nd);
                    }
                }
            });
        IpplTimings::stopTimer(outer_loop);

        if (bcType == PERIODIC_FACE) {
            resultField.accumulateHalo();
            bcField.apply(resultField);
            bcField.assignGhostToPhysical(resultField);
        } else {
            resultField.accumulateHalo_noghost();
        }

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_upperlower(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get boundary conditions from field
        BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType = bcField[0]->getBCType();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex                            = elementIndices(index);
                const Vector<size_t, numElementDOFs> local_dof = this->getLocalDOFIndices();
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices
                size_t i, j;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Handle boundary DOFs
                    // If Zero Dirichlet BCs, skip this DOF
                    // If Constant Dirichlet BCs, identity
                    if ((bcType == CONSTANT_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                        }
                        apply(resultView, I_nd) =  apply(view, I_nd);
                        continue;
                    } else if ((bcType == ZERO_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }

                    for (j = 0; j < numElementDOFs; ++j) {
                        J_nd = global_dof_ndindices[j];

                        if (global_dofs[i] == global_dofs[j]) {
                            continue;
                        }

                        // Skip boundary DOFs (Zero & Constant Dirichlet BCs)
                        if (((bcType == ZERO_FACE) || (bcType == CONSTANT_FACE)) 
                            && this->isDOFOnBoundary(J_nd)) {
                            continue;
                        }

                        // get the appropriate index for the Kokkos view of the field
                        for (unsigned d = 0; d < Dim; ++d) {
                            J_nd[d] = J_nd[d] - ldom[d].first() + nghost;
                        }

                        apply(resultView, I_nd) += A_K[i][j] * apply(view, J_nd);
                    }
                }
            });
        IpplTimings::stopTimer(outer_loop);

        if (bcType == PERIODIC_FACE) {
            resultField.accumulateHalo();
            bcField.apply(resultField);
            bcField.assignGhostToPhysical(resultField);
        } else {
            resultField.accumulateHalo_noghost();
        }

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_inversediag(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get boundary conditions from field
        BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType = bcField[0]->getBCType();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex                            = elementIndices(index);
                const Vector<size_t, numElementDOFs> local_dof = this->getLocalDOFIndices();
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices
                size_t i, j;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Handle boundary DOFs
                    // If Zero Dirichlet BCs, skip this DOF
                    // If Constant Dirichlet BCs, identity
                    if ((bcType == CONSTANT_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                        }
                        apply(resultView, I_nd) = 1.0;
                        continue;
                    } else if ((bcType == ZERO_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }

                    for (j = 0; j < numElementDOFs; ++j) {
                        if (global_dofs[i] == global_dofs[j]) {
                            J_nd = global_dof_ndindices[j];

                            // Skip boundary DOFs (Zero & Constant Dirichlet BCs)
                            if (((bcType == ZERO_FACE) || (bcType == CONSTANT_FACE)) 
                                && this->isDOFOnBoundary(J_nd)) {
                                continue;
                            }

                            // get the appropriate index for the Kokkos view of the field
                            for (unsigned d = 0; d < Dim; ++d) {
                                J_nd[d] = J_nd[d] - ldom[d].first() + nghost;
                            }

                            // sum up all contributions of element matrix
                            apply(resultView, I_nd) += A_K[i][j];
                        }
                    }
                }
            });

        if (bcType == PERIODIC_FACE) {
            resultField.accumulateHalo();
            bcField.apply(resultField);
            bcField.assignGhostToPhysical(resultField);
        } else {
            resultField.accumulateHalo_noghost();
        }

        // apply the inverse diagonal after already summed all contributions from element matrices
        using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;
        ippl::parallel_for("Loop over result view to apply inverse", field.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const index_array_type& args) {
                if (apply(resultView, args) != 0.0) {
                    apply(resultView, args) = (1.0 / apply(resultView, args)) * apply(view, args);
                }
            });
        IpplTimings::stopTimer(outer_loop);

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_diag(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get boundary conditions from field
        BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType = bcField[0]->getBCType();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex                            = elementIndices(index);
                const Vector<size_t, numElementDOFs> local_dof = this->getLocalDOFIndices();
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices
                size_t i, j;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Handle boundary DOFs
                    // If Zero Dirichlet BCs, skip this DOF
                    // If Constant Dirichlet BCs, identity
                    if ((bcType == CONSTANT_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        for (unsigned d = 0; d < Dim; ++d) {
                            I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                        }
                        apply(resultView, I_nd) =  apply(view, I_nd);
                        continue;
                    } else if ((bcType == ZERO_FACE) && (this->isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }

                    for (j = 0; j < numElementDOFs; ++j) {
                        if (global_dofs[i] == global_dofs[j]) {
                            J_nd = global_dof_ndindices[j];

                            // Skip boundary DOFs (Zero & Constant Dirichlet BCs)
                            if (((bcType == ZERO_FACE) || (bcType == CONSTANT_FACE)) 
                                && this->isDOFOnBoundary(J_nd)) {
                                continue;
                            }

                            // get the appropriate index for the Kokkos view of the field
                            for (unsigned d = 0; d < Dim; ++d) {
                                J_nd[d] = J_nd[d] - ldom[d].first() + nghost;
                            }

                            apply(resultView, I_nd) += A_K[i][j] * apply(view, J_nd);
                        }
                    }
                }
            });
        IpplTimings::stopTimer(outer_loop);

        if (bcType == PERIODIC_FACE) {
            resultField.accumulateHalo();
            bcField.apply(resultField);
            bcField.assignGhostToPhysical(resultField);
        } else {
            resultField.accumulateHalo_noghost();
        }

        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                           FieldRHS>::evaluateAx_lift(FieldLHS& field, F& evalFunction) const {
        Inform m("");

        // get number of ghost cells in field
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> grad_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        // Make local element matrix -- does not change through the element mesh
        // Element matrix
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A_K;

        // 1. Compute the Galerkin element matrix A_K
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A_K[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                }
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = field.getView();
        AtomicViewType resultView = resultField.getView();

        // Get domain information
        auto ldom = (field.getLayout()).getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex                            = elementIndices(index);
                const Vector<size_t, numElementDOFs> local_dof = this->getLocalDOFIndices();
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);
                Vector<indices_t, numElementDOFs> global_dof_ndindices;

                for (size_t i = 0; i < numElementDOFs; ++i) {
                    global_dof_ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);
                }

                // local DOF indices (both i and j go from 0 to numDOFs-1 in the element)
                size_t i, j;

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Skip if on a row of the matrix
                    if (this->isDOFOnBoundary(I_nd)) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }

                    for (j = 0; j < numElementDOFs; ++j) {
                        J_nd = global_dof_ndindices[j];

                        // Contribute to lifting only if on a boundary DOF
                        if (this->isDOFOnBoundary(J_nd)) {
                            // get the appropriate index for the Kokkos view of the field
                            for (unsigned d = 0; d < Dim; ++d) {
                                J_nd[d] = J_nd[d] - ldom[d].first() + nghost;
                            }
                            apply(resultView, I_nd) += A_K[i][j] * apply(view, J_nd);
                            continue;
                        }

                    }
                }
            });
        resultField.accumulateHalo();

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                       FieldRHS>::evaluateLoadVector(FieldRHS& field) const {
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
        Vector<Vector<T, numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Get domain information and ghost cells
        auto ldom        = (field.getLayout()).getLocalNDIndex();
        const int nghost = field.getNghost();

        // Get boundary conditions from field
        BConds<FieldRHS, Dim>& bcField = field.getFieldBC();
        FieldBC bcType = bcField[0]->getBCType();

        FieldRHS temp_field(field.get_mesh(), field.getLayout(), nghost);
        temp_field.setFieldBC(bcField);

        // Get field data and make it atomic,
        // since it will be added to during the kokkos loop
        // We work with a temporary field since we need to use field
        // to evaluate the load vector; then we assign temp to RHS field
        AtomicViewType atomic_view = temp_field.getView();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // start a timer
        static IpplTimings::TimerRef outer_loop =
            IpplTimings::getTimer("evaluateLoadVec: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(size_t index) {
                const size_t elementIndex                              = elementIndices(index);
                const Vector<size_t, numElementDOFs> local_dofs  = this->getLocalDOFIndices();
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);

                size_t i, I;

                // 1. Compute b_K
                for (i = 0; i < numElementDOFs; ++i) {
                    I = global_dofs[i];

                    // TODO fix for higher order
                    auto dof_ndindex_I = this->getMeshVertexNDIndex(I);

                    // Skip boundary DOFs (Zero and Constant Dirichlet BCs)
                    if (((bcType == ZERO_FACE) || (bcType == CONSTANT_FACE))
                        && (this->isDOFOnBoundary(dof_ndindex_I))) {
                        continue;
                    }

                    // calculate the contribution of this element
                    T contrib = 0;
                    for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                        T val = 0;
                        for (size_t j = 0; j < numElementDOFs; ++j) {
                            // get field index corresponding to this DOF
                            size_t J           = global_dofs[j];
                            auto dof_ndindex_J = this->getMeshVertexNDIndex(J);
                            for (unsigned d = 0; d < Dim; ++d) {
                                dof_ndindex_J[d] = dof_ndindex_J[d] - ldom[d].first() + nghost;
                            }

                            // get field value at DOF and interpolate to q_k
                            val += basis_q[k][j] * apply(field, dof_ndindex_J);
                        }

                        contrib += w[k] * basis_q[k][i] * absDetDPhi * val;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        dof_ndindex_I[d] = dof_ndindex_I[d] - ldom[d].first() + nghost;
                    }

                    // add the contribution of the element to the field
                    apply(atomic_view, dof_ndindex_I) += contrib;

                }
            });
        IpplTimings::stopTimer(outer_loop);

        temp_field.accumulateHalo();

        if ((bcType == PERIODIC_FACE) || (bcType == CONSTANT_FACE)) {
            bcField.apply(temp_field);
            bcField.assignGhostToPhysical(temp_field);
        }

        field = temp_field;

        IpplTimings::stopTimer(evalLoadV);
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    T LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::computeErrorL2(
        const FieldLHS& u_h, const F& u_sol) const {
        if (this->quadrature_m.getOrder() < (2 * Order + 1)) {
            // throw exception
            throw IpplException(
                "LagrangeSpace::computeErrorL2()",
                "Order of quadrature rule for error computation should be > 2*p + 1");
        }

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
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
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);

                // contribution of this element to the error
                T contrib = 0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    T val_u_sol = u_sol(this->ref_element_m.localToGlobal(
                        this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex)),
                        q[k]));

                    T val_u_h = 0;
                    for (size_t i = 0; i < numElementDOFs; ++i) {
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
    T LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::computeAvg(
        const FieldLHS& u_h) const {
        if (this->quadrature_m.getOrder() < (2 * Order + 1)) {
            // throw exception
            throw IpplException(
                "LagrangeSpace::computeAvg()",
                "Order of quadrature rule for error computation should be > 2*p + 1");
        }

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        const indices_t zeroNdIndex = Vector<size_t, Dim>(0);

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Variable to sum the error to
        T avg = 0;

        // Get domain information and ghost cells
        auto ldom        = (u_h.getLayout()).getLocalNDIndex();
        const int nghost = u_h.getNghost();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // Loop over elements to compute contributions
        Kokkos::parallel_reduce(
            "Compute average over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(size_t index, double& local) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);

                // contribution of this element to the error
                T contrib = 0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    T val_u_h = 0;
                    for (size_t i = 0; i < numElementDOFs; ++i) {
                        // get field index corresponding to this DOF
                        size_t I           = global_dofs[i];
                        auto dof_ndindex_I = this->getMeshVertexNDIndex(I);
                        for (unsigned d = 0; d < Dim; ++d) {
                            dof_ndindex_I[d] = dof_ndindex_I[d] - ldom[d].first() + nghost;
                        }

                        // get field value at DOF and interpolate to q_k
                        val_u_h += basis_q[k][i] * apply(u_h, dof_ndindex_I);
                    }

                    contrib += w[k] * val_u_h * absDetDPhi;
                }
                local += contrib;
            },
            Kokkos::Sum<double>(avg));

        // MPI reduce
        T global_avg = 0.0;
        Comm->allreduce(avg, global_avg, 1, std::plus<T>());

        return global_avg;
    }

}  // namespace ippl
