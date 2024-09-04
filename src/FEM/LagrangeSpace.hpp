
namespace ippl {
    template <typename T, unsigned Dim>
    KOKKOS_FUNCTION T sinusoidalRHSFunction(ippl::Vector<T, Dim> x_vec) {
        const T pi = Kokkos::numbers::pi_v<T>;

        T val = 1.0;
        for (unsigned d = 0; d < Dim; d++) {
            val *= Kokkos::sin(pi * x_vec[d]);
        }

        return Dim * pi * pi * val;
    }

    // LagrangeSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType, 
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::LagrangeSpace(
        const Mesh<T, Dim>& mesh,
        const ElementType& ref_element,
        const QuadratureType& quadrature,
        const Layout_t& layout)
        : FiniteElementSpace<T, Dim, getLagrangeNumElementDOFs(Dim, Order), ElementType, QuadratureType,
                             FieldLHS, FieldRHS>(mesh, ref_element, quadrature) {
        // Assert that the dimension is either 1, 2 or 3.
        static_assert(Dim >= 1 && Dim <= 3,
                      "Finite Element space only supports 1D, 2D and 3D meshes");

        // Initialize the elementIndices view
        initializeElementIndices(layout);
    }

    // Initialize element indices Kokkos View
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::initializeElementIndices(const Layout_t& layout) {
        const auto& ldom = layout.getLocalNDIndex(); 
        int npoints = ldom.size();
        auto first  = ldom.first();
        auto last   = ldom.last();
        ippl::Vector<double, Dim> bounds;

        for (size_t d = 0; d < Dim; ++d) {
            bounds[d] = this->nr_m[d] - 1;
        }

        int upperBoundaryPoints = -1;

        Kokkos::View<size_t*> points("ComputeMapping", npoints);
        Kokkos::parallel_reduce("ComputePoints", npoints,
            KOKKOS_CLASS_LAMBDA(const int i, int& local) {
                int idx = i;
                ndindex_t val;
                bool isBoundary = false;
                for (unsigned int d = 0; d < Dim; ++d) {
                    int range = last[d] - first[d] + 1;
                    val[d] = first[d] + (idx % range);
                    idx /= range;
                    if (val[d] == bounds[d]) {
                        isBoundary = true;
                    }
                }
                points(i) = (!isBoundary) * (this->getElementIndex(val));
                local += isBoundary;
            }, Kokkos::Sum<int>(upperBoundaryPoints));
        Kokkos::fence();

        int elementsPerRank = npoints - upperBoundaryPoints;
        elementIndices      = Kokkos::View<size_t*>("i", elementsPerRank);
        Kokkos::View<size_t> index("index");

        Kokkos::parallel_for("RemoveNaNs", npoints,
            KOKKOS_CLASS_LAMBDA(const int i) {
                if ((points(i) != 0) || (i == 0)) {
                    const size_t idx = Kokkos::atomic_fetch_add(&index(), 1);
                    elementIndices(idx) = points(i);
                }
            });

        // naive implementation below
        /*
        
        const size_t numElements = this->numElements();
        const unsigned int numRanks   = Comm->size();
        //const unsigned int myRank     = Comm->rank();
        size_t elementsPerRank   = numElements/numRanks;
        unsigned int remainder        = numElements - (elementsPerRank * numRanks);
        // if elements are remaining to be assigned to a rank, assign them
        if (myRank < remainder) {
            elementsPerRank++;
        }

        elementIndices = Kokkos::View<size_t*>("i", elementsPerRank);

        for (size_t i = 0; i < elementsPerRank; ++i) {
            size_t global = i + ldom[0].first();
            myelements.push_back(global);
        }

        using exec_space  = typename Kokkos::View<size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;
        
        Kokkos::parallel_for("Element index view", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t i) {
                size_t global = i + ldom[0].first();
                elementIndices(i) = global;
            });
        Kokkos::fence();
        */
    }

    ///////////////////////////////////////////////////////////////////////
    /// Degree of Freedom operations //////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, 
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    size_t LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::numGlobalDOFs()
        const {
        size_t num_global_dofs = 1;
        for (size_t d = 0; d < Dim; ++d) {
            num_global_dofs *= this->nr_m[d] * Order;
        }

        return num_global_dofs;
    }

    /*
    template <typename T, unsigned Dim, unsigned Order, typename QuadratureType, typename FieldLHS,
              typename FieldRHS>
    KOKKOS_FUNCTION
    typename LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t
    LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::getLocalDOFIndex(
        const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t&
            elementIndex,
        const LagrangeSpace<T, Dim, Order, QuadratureType, FieldLHS, FieldRHS>::index_t&
            globalDOFIndex) const {
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
            throw IpplException("LagrangeSpace::getLocalDOFIndex()", "FEM Lagrange Space: Dimension not supported");
        }

        // Find the global DOF in the vector and return the local DOF index
        // TODO this can be done faster since the global DOFs are sorted
        for (index_t i = 0; i < dof_mapping.dim; ++i) {
            if (global_dofs[dof_mapping[i]] == globalDOFIndex) {
                return dof_mapping[i];
            }
        }
        throw IpplException("LagrangeSpace::getLocalDOFIndex()", "FEM Lagrange Space: Global DOF not found in specified element");
    }
    */

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    typename LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::index_t
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::getGlobalDOFIndex(
        const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::index_t&
            elementIndex,
        const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::index_t&
            localDOFIndex) const {
        const auto global_dofs = this->getGlobalDOFIndices(elementIndex);

        return global_dofs[localDOFIndex];
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    Vector<typename LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::index_t,
           LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::getLocalDOFIndices() const {
        Vector<index_t, numElementDOFs> localDOFs;

        for (index_t dof = 0; dof < numElementDOFs; ++dof) {
            localDOFs[dof] = dof;
        }

        return localDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, 
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    Vector<typename LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::index_t,
           LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::getGlobalDOFIndices(
        const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::index_t&
            elementIndex) const {
        Vector<index_t, this->numElementDOFs> globalDOFs(0);

        // get element pos
        ndindex_t elementPos = this->getElementNDIndex(elementIndex);

        // Compute the vector to multiply the ndindex with
        ippl::Vector<size_t, Dim> vec(1);
        for (size_t d = 1; d < dim; ++d) {
            for (size_t d2 = d; d2 < Dim; ++d2) {
                vec[d2] *= this->nr_m[d - 1];
            }
        }
        vec *= Order;  // Multiply each dimension by the order
        index_t smallestGlobalDOF = elementPos.dot(vec);

        // Add the vertex DOFs
        globalDOFs[0] = smallestGlobalDOF;
        globalDOFs[1] = smallestGlobalDOF + Order;

        if (Dim >= 2) {
            globalDOFs[2] = globalDOFs[1] + this->nr_m[1] * Order;
            globalDOFs[3] = globalDOFs[0] + this->nr_m[1] * Order;
        }
        if (Dim >= 3) {
            globalDOFs[4] =
                globalDOFs[0] + this->nr_m[1] * this->nr_m[2] * Order;
            globalDOFs[5] =
                globalDOFs[1] + this->nr_m[1] * this->nr_m[2] * Order;
            globalDOFs[6] =
                globalDOFs[2] + this->nr_m[1] * this->nr_m[2] * Order;
            globalDOFs[7] =
                globalDOFs[3] + this->nr_m[1] * this->nr_m[2] * Order;
        }

        if (Order > 1) {
            // If the order is greater than 1, there are edge and face DOFs, otherwise the work is
            // done

            // Add the edge DOFs
            if (Dim >= 2) {
                for (index_t i = 0; i < Order - 1; ++i) {
                    globalDOFs[8 + i] = globalDOFs[0] + i + 1;
                    globalDOFs[8 + Order - 1 + i] =
                        globalDOFs[1] + (i + 1) * this->nr_m[1];
                    globalDOFs[8 + 2 * (Order - 1) + i] = globalDOFs[2] - (i + 1);
                    globalDOFs[8 + 3 * (Order - 1) + i] =
                        globalDOFs[3] - (i + 1) * this->nr_m[1];
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
                            globalDOFs[0] + (i + 1) + (j + 1) * this->nr_m[1];
                        globalDOFs[8 + 4 * (Order - 1) + (Order - 1) * (Order - 1) + i * (Order - 1)
                                   + j] =
                            globalDOFs[1] + (i + 1) + (j + 1) * this->nr_m[1];
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

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    Vector<typename LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::ndindex_t,
           LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::numElementDOFs>
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::getGlobalDOFNDIndices(
        const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::index_t&
            elementIndex) const {
        static_assert(Order == 1 && "Only order 1 is supported at the moment");

        Vector<ndindex_t, numElementDOFs> ndindices;

        // 1. get all the global DOFs for the element
        Vector<index_t, numElementDOFs> global_dofs = this->getGlobalDOFIndices(elementIndex);

        // 2. convert the global DOFs to ndindices
        for (index_t i = 0; i < numElementDOFs; ++i) {
            ndindices[i] = this->getMeshVertexNDIndex(global_dofs[i]);  // TODO fix for higher order
        }

        return ndindices;
    }

    ///////////////////////////////////////////////////////////////////////
    /// Basis functions and gradients /////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, 
              typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    T LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        evaluateRefElementShapeFunction(const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType,
                                                            FieldLHS, FieldRHS>::index_t& localDOF,
                                        const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, 
                                                            FieldLHS, FieldRHS>::point_t& localPoint) const {
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

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType,
              typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    typename LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::gradient_vec_t
    LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::
        evaluateRefElementShapeFunctionGradient(
            const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::index_t&
                localDOF,
            const LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::point_t&
                localPoint) const {
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

    ///////////////////////////////////////////////////////////////////////
    /// Assembly operations ///////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType,
              typename FieldLHS, typename FieldRHS>
    FieldLHS LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::evaluateAx(
        const FieldLHS& field,
        const std::function<T(
            const index_t&, const index_t&,
            const Vector<Vector<T, Dim>, LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,
                                                       FieldRHS>::numElementDOFs>&)>& evalFunction)
        const {

        Inform m("");
        m << "inside evalAx, start" << endl;

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);

        // get number of ghost cells in field 
        const int nghost = field.getNghost();

        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FieldLHS resultField(field.get_mesh(), field.getLayout(), nghost);

        bool checkEssentialBDCs = true;  // TODO get from field in the future
        // T bc_const_value        = 1.0;   // TODO get from field (non-homogeneous BCs)

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        Vector<Vector<gradient_vec_t, this->numElementDOFs>, QuadratureType::numElementNodes>
            grad_b_q;
        for (index_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (index_t i = 0; i < this->numElementDOFs; ++i) {
                grad_b_q[k][i] = this->evaluateRefElementShapeFunctionGradient(i, q[k]);
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

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for("Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const index_t elementIndex                                         = elementIndices(index);
                const Vector<index_t, this->numElementDOFs> local_dofs             = this->getLocalDOFIndices();
                const Vector<ndindex_t, this->numElementDOFs> global_dof_ndindices = this->getGlobalDOFNDIndices(elementIndex);

                // local DOF indices
                index_t i, j;

                // Element matrix
                Vector<Vector<T, this->numElementDOFs>, this->numElementDOFs> A_K;

                // 1. Compute the Galerkin element matrix A_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    for (j = 0; j < this->numElementDOFs; ++j) {
                        A_K[i][j] = 0.0;
                        for (index_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                            A_K[i][j] += w[k] * evalFunction(i, j, grad_b_q[k]);
                        }
                    }
                }

                // global DOF n-dimensional indices (Vector of N indices representing indices in each
                // dimension)
                ndindex_t I_nd, J_nd;

                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Skip boundary DOFs (Zero Dirichlet BCs)
                    if (checkEssentialBDCs && this->isDOFOnBoundary(I_nd)) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + nghost;
                    }

                    for (j = 0; j < this->numElementDOFs; ++j) {
                        J_nd = global_dof_ndindices[j];

                        // Skip boundary DOFs (Zero Dirichlet BCs)
                        if (checkEssentialBDCs && this->isDOFOnBoundary(J_nd)) {
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
        IpplTimings::stopTimer(evalAx);

        return resultField;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType,
              typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    T LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::evalFunc(
        const T absDetDPhi,
        const index_t elementIndex, const index_t& i, const point_t& q_k,
        const Vector<T, numElementDOFs>& basis_q_k) const {

        Vector<T, Dim> coords = this->ref_element_m.localToGlobal(
                                this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex)), q_k);

        const T& f_q_k = sinusoidalRHSFunction<T,Dim>(coords);
        
        return f_q_k * basis_q_k[i] * absDetDPhi;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType,
              typename FieldLHS, typename FieldRHS>
    void LagrangeSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::evaluateLoadVector(
        FieldRHS& field) const {

        Inform m("");
        m << "inside evalLoadVec, start" << endl;

        // start a timer
        static IpplTimings::TimerRef evalLoadV = IpplTimings::getTimer("evaluateLoadVector");
        IpplTimings::startTimer(evalLoadV);

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        const ndindex_t zeroNdIndex = Vector<index_t, Dim>(0);

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<T, this->numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (index_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (index_t i = 0; i < this->numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        // Absolute value of det Phi_K
        const T absDetDPhi = std::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        m << "get abs det phi = " << absDetDPhi << endl;

        // Get field data and make it atomic,
        // since it will be added to during the kokkos loop
        AtomicViewType atomic_view = field.getView();

        // Get domain information and ghost cells
        auto ldom          = (field.getLayout()).getLocalNDIndex();
        const int nghost   = field.getNghost();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        auto const&reference_element = this->ref_element_m;

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateLoadVec: outer loop");
        IpplTimings::startTimer(outer_loop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for("Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(size_t index) {
                const index_t elementIndex                              = elementIndices(index);
                const Vector<index_t, this->numElementDOFs> local_dofs  = this->getLocalDOFIndices();
                const Vector<index_t, this->numElementDOFs> global_dofs = this->getGlobalDOFIndices(elementIndex);

                printf("inside kokkos loop!\n");

                index_t i, I;

                // 1. Compute b_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    I = global_dofs[i];

                    // TODO fix for higher order
                    auto dof_ndindex_I = this->getMeshVertexNDIndex(I);

                    // homogeneous Dirichlet boundary conditions
                    if (this->isDOFOnBoundary(dof_ndindex_I)) {
                        continue;
                    }

                    // calculate the contribution of this element
                    T contrib = 0;
                    for (index_t k = 0; k < QuadratureType::numElementNodes; ++k) {

                        printf("before anything \n");

                        auto points = this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex));

                        printf("got points \n");
                        
                        point_t val_in = this->ref_element_m.localToGlobal(points, q[k]);
                        
                        printf("got val_in \n");

                        T val = sinusoidalRHSFunction<T,Dim>(val_in);

                        //T val = sinusoidalRHSFunction<T,Dim>(this->ref_element_m->localToGlobal(
                        //    this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex)), q[k]));
                        
                        printf("after val");

                        contrib += w[k] * basis_q[k][i] * absDetDPhi * val;
                    }
                    printf("after contribution computation");
                    
                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        dof_ndindex_I[d] = dof_ndindex_I[d] - ldom[d].first() + nghost;
                    }

                    printf("before apply()");

                    // add the contribution of the element to the field
                    apply(atomic_view, dof_ndindex_I) += contrib;

                    printf("after assign to apply");

                }
        });
        IpplTimings::stopTimer(outer_loop);
        IpplTimings::stopTimer(evalLoadV);

        Kokkos::fence();
        field.write();
        Kokkos::fence();

        m << "inside evalLoadVec, end" << endl;

    }

}  // namespace ippl
