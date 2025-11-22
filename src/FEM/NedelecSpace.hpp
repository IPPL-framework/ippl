#include "NedelecSpace.h"
namespace ippl {

    // NedelecSpace constructor, which calls the FiniteElementSpace constructor,
    // and decomposes the elements among ranks according to layout.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::NedelecSpace(UniformCartesian<T, Dim>& mesh, ElementType& ref_element,
                                const QuadratureType& quadrature, const Layout_t& layout)
                            : FiniteElementSpace<T, Dim, getNedelecNumElementDOFs(Dim, Order),
                                ElementType, QuadratureType, FEMVector<T>, FEMVector<T>>
                                    (mesh, ref_element, quadrature) {
        // Assert that the dimension is either 2 or 3.
        static_assert(Dim >= 2 && Dim <= 3,
            "The Nedelec Finite Element space only supports 2D and3D meshes");

        // Initialize the elementIndices view
        initializeElementIndices(layout);
    }

    // NedelecSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::NedelecSpace(UniformCartesian<T, Dim>& mesh, ElementType& ref_element,
                                const QuadratureType& quadrature)
                            : FiniteElementSpace<T, Dim, getNedelecNumElementDOFs(Dim, Order),
                            ElementType, QuadratureType, FEMVector<T>, FEMVector<T>>
                                (mesh, ref_element, quadrature) {

        // Assert that the dimension is either 2 or 3.
        static_assert(Dim >= 2 && Dim <= 3,
            "The Nedelec Finite Element space only supports 2D and 3D meshes");
    }

    // NedelecSpace initializer, to be made available to the FEMPoissonSolver 
    // such that we can call it from setRhs.
    // Sets the correct mesh and decomposes the elements among ranks according to layout.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    void NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::initialize(UniformCartesian<T, Dim>& mesh, const Layout_t& layout) {
        
        FiniteElementSpace<T, Dim, getNedelecNumElementDOFs(Dim, Order),
                            ElementType, QuadratureType, FEMVector<T>, FEMVector<T>>::setMesh(mesh);

        // Initialize the elementIndices view
        initializeElementIndices(layout);

        // set the local DOF position vector
        localDofPositions_m(0)(0) = 0.5; 
        localDofPositions_m(1)(1) = 0.5;
        localDofPositions_m(2)(0) = 0.5; localDofPositions_m(2)(1) = 1;
        localDofPositions_m(3)(0) = 1;   localDofPositions_m(3)(1) = 0.5;
        localDofPositions_m(4)(2) = 0.5;
        localDofPositions_m(5)(0) = 1;   localDofPositions_m(5)(2) = 0.5;
        localDofPositions_m(6)(0) = 1;   localDofPositions_m(6)(1) = 1;
            localDofPositions_m(6)(2) = 0.5;
        localDofPositions_m(7)(1) = 1;   localDofPositions_m(7)(2) = 0.5;
        localDofPositions_m(8)(0) = 0.5; localDofPositions_m(8)(2) = 1;
        localDofPositions_m(9)(1) = 0.5; localDofPositions_m(9)(2) = 1;
        localDofPositions_m(10)(0) = 0.5; localDofPositions_m(10)(1) = 1;
            localDofPositions_m(10)(2) = 1;
        localDofPositions_m(11)(0) = 1;   localDofPositions_m(11)(1) = 0.5;
            localDofPositions_m(11)(2) = 1;
    }

    // Initialize element indices Kokkos View
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    void NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::initializeElementIndices(const Layout_t& layout) {
        
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
            }
        );
    }

    ///////////////////////////////////////////////////////////////////////
    /// Degree of Freedom operations //////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION size_t NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::numGlobalDOFs() const {

        size_t num_global_dofs = 0;
        
        for (size_t d = 0; d < Dim; ++d) {
            size_t accu = this->nr_m[d]-1;
            for (size_t d2 = 0; d2 < Dim; ++d2) {
                if (d == d2) continue;
                accu *= this->nr_m[d2];
            }
            num_global_dofs += accu;
        }

        return num_global_dofs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION size_t NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::getLocalDOFIndex(const size_t& elementIndex,
                                    const size_t& globalDOFIndex) const {

        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");

        // Get all the global DOFs for the element
        const Vector<size_t, numElementDOFs> global_dofs =
            this->NedelecSpace::getGlobalDOFIndices(elementIndex);

        ippl::Vector<size_t, numElementDOFs> dof_mapping;
        if (Dim == 2) {
            dof_mapping = {0, 1, 2, 3};
        } else if (Dim == 3) {
            dof_mapping = {0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 10, 11};
        }

        // Find the global DOF in the vector and return the local DOF index
        // TODO this can be done faster since the global DOFs are sorted
        for (size_t i = 0; i < dof_mapping.dim; ++i) {
            if (global_dofs[dof_mapping[i]] == globalDOFIndex) {
                return dof_mapping[i];
            }
        }
        // it would be good to throw an error in this case
        // just like the comment in the LagrangeSpace::getLocalDOFIndex()
        return 0;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION size_t NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::getGlobalDOFIndex(const size_t& elementIndex,
                                const size_t& localDOFIndex) const {

        const auto global_dofs = this->NedelecSpace::getGlobalDOFIndices(elementIndex);

        return global_dofs[localDOFIndex];
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION Vector<size_t, NedelecSpace<T, Dim, Order, ElementType,
                        QuadratureType, FieldType>::numElementDOFs>
                    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::getLocalDOFIndices() const {

        Vector<size_t, numElementDOFs> localDOFs;

        for (size_t dof = 0; dof < numElementDOFs; ++dof) {
            localDOFs[dof] = dof;
        }

        return localDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION Vector<size_t, NedelecSpace<T, Dim, Order, ElementType,
                        QuadratureType, FieldType>::numElementDOFs>
                    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::getGlobalDOFIndices(const NedelecSpace<T, Dim, Order, ElementType,
                                QuadratureType, FieldType>::indices_t& elementIndex) const {

        
        // These are simply some manual caclualtions that need to be done.
        
        Vector<size_t, numElementDOFs> globalDOFs(0);

        // Initialize a helper vector v
        Vector<size_t, Dim> v(1);
        if constexpr (Dim == 2) {
            size_t nx = this->nr_m[0];
            v(1) = 2*nx-1;
        } else if constexpr (Dim == 3) {
            size_t nx = this->nr_m[0];
            size_t ny = this->nr_m[1];
            v(1) = 2*nx -1;
            v(2) = 3*nx*ny - nx - ny;
        }

        // For both 2D and 3D the first few DOF indices are the same
        size_t nx = this->nr_m[0];
        globalDOFs(0) = v.dot(elementIndex);
        globalDOFs(1) = globalDOFs(0) + nx - 1;
        globalDOFs(2) = globalDOFs(1) + nx;
        globalDOFs(3) = globalDOFs(1) + 1;

        if constexpr (Dim == 3) {
            size_t ny = this->nr_m[1];

            globalDOFs(4) = v(2)*elementIndex(2) + 2*nx*ny - nx - ny
                + elementIndex(1)*nx + elementIndex(0);
            globalDOFs(5) = globalDOFs(4) + 1;
            globalDOFs(6) = globalDOFs(4) + nx + 1;
            globalDOFs(7) = globalDOFs(4) + nx;
            globalDOFs(8) = globalDOFs(0) + 3*nx*ny - nx - ny;
            globalDOFs(9) = globalDOFs(8) + nx - 1;
            globalDOFs(10) = globalDOFs(9) + nx;
            globalDOFs(11) = globalDOFs(9) + 1;
        }
        

        return globalDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION Vector<size_t, NedelecSpace<T, Dim, Order, ElementType,
                        QuadratureType, FieldType>::numElementDOFs>
                    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::getGlobalDOFIndices(const size_t& elementIndex) const {

        indices_t elementPos = this->getElementNDIndex(elementIndex);
        return getGlobalDOFIndices(elementPos);
    }

    

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION Vector<size_t, NedelecSpace<T, Dim, Order, ElementType,
                        QuadratureType, FieldType>::numElementDOFs>
                    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::getFEMVectorDOFIndices(NedelecSpace<T, Dim, Order, ElementType,
                                QuadratureType, FieldType>::indices_t elementIndex,
                                NDIndex<Dim> ldom) const {
                            

        // This function here is pretty much the same as getGlobalDOFIndices()
        // the only difference is the domain size and that we have an offset
        // of the subdomain of the rank to the global one, else it is the same

        Vector<size_t, numElementDOFs> FEMVectorDOFs(0);
        
        // This corresponds to translating a global element position to one in
        // the subdomain of the rank. For this we subtract the starting position
        // the rank subdomain and add the "ghost" hyperplane.
        elementIndex -= ldom.first();
        elementIndex += 1;
        
        indices_t dif(0);
        dif = ldom.last() - ldom.first();
        dif += 1 + 2; // plus 1 for last still being in +2 for ghosts.
        
        // From here on out it is pretty much the same as the
        // getGlobalDOFIndices() function.
        Vector<size_t, Dim> v(1);
        if constexpr (Dim == 2) {
            size_t nx = dif[0];
            v(1) = 2*nx-1;
        } else if constexpr (Dim == 3) {
            size_t nx = dif[0];
            size_t ny = dif[1];
            v(1) = 2*nx -1;
            v(2) = 3*nx*ny - nx - ny;
        }

        size_t nx = dif[0];
        FEMVectorDOFs(0) = v.dot(elementIndex);
        FEMVectorDOFs(1) = FEMVectorDOFs(0) + nx - 1;
        FEMVectorDOFs(2) = FEMVectorDOFs(1) + nx;
        FEMVectorDOFs(3) = FEMVectorDOFs(1) + 1;

        if constexpr (Dim == 3) {
            size_t ny = dif[1];

            FEMVectorDOFs(4) = v(2)*elementIndex(2) + 2*nx*ny - nx - ny
                + elementIndex(1)*nx + elementIndex(0);
            FEMVectorDOFs(5) = FEMVectorDOFs(4) + 1;
            FEMVectorDOFs(6) = FEMVectorDOFs(4) + nx + 1;
            FEMVectorDOFs(7) = FEMVectorDOFs(4) + nx;
            FEMVectorDOFs(8) = FEMVectorDOFs(0) + 3*nx*ny - nx - ny;
            FEMVectorDOFs(9) = FEMVectorDOFs(8) + nx - 1;
            FEMVectorDOFs(10) = FEMVectorDOFs(9) + nx;
            FEMVectorDOFs(11) = FEMVectorDOFs(9) + 1;
        }
        
        
        return FEMVectorDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION Vector<size_t, NedelecSpace<T, Dim, Order, ElementType,
                        QuadratureType, FieldType>::numElementDOFs>
                    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::getFEMVectorDOFIndices(const size_t& elementIndex, NDIndex<Dim> ldom) const {
        
        // First get the global element position
        indices_t elementPos = this->getElementNDIndex(elementIndex);
        return getFEMVectorDOFIndices(elementPos, ldom);
    }


    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION typename NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>::point_t
        NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
            ::getLocalDOFPosition(size_t localDOFIndex) const {
        
        // Hardcoded center of edges which are stored in the localDofPositions_m
        // vector. If the DOF position of an edge element actually is the center
        // of the edge is a different question...
        return localDofPositions_m(localDOFIndex);
    }



    ///////////////////////////////////////////////////////////////////////
    /// Assembly operations ///////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
                typename QuadratureType, typename FieldType>
    template <typename F>
    FEMVector<T> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::evaluateAx(FEMVector<T>& x, F& evalFunction) const {
        Inform m("");
        

        IpplTimings::TimerRef timerAxInit = IpplTimings::getTimer("Ax init");
        IpplTimings::startTimer(timerAxInit);

        // create a new field for result, default initialized to zero thanks to
        // the Kokkos::View
        FEMVector<T> resultVector = x.template skeletonCopy<T>();

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();
        
        // Get the values of the basis functions and their curl at the
        // quadrature points.
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> curl_b_q;
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> val_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                curl_b_q[k][i] = this->evaluateRefElementShapeFunctionCurl(i, q[k]);
                val_b_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        // Get field data and atomic result data,
        // since it will be added to during the kokkos loop
        ViewType view             = x.getView();
        AtomicViewType resultView = resultVector.getView();


        // Get domain information
        auto ldom = layout_m.getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;


        IpplTimings::stopTimer(timerAxInit);

        // Here we assemble the local matrix of an element. In theory this would
        // have to be done for each element individually, but because we have
        // that in the case of IPPL all the elements have the same shape we can
        // also just do it once and then use if all the time.
        IpplTimings::TimerRef timerAxLocalMatrix = IpplTimings::getTimer("Ax local matrix");
        IpplTimings::startTimer(timerAxLocalMatrix);
        Vector<Vector<T, numElementDOFs>, numElementDOFs> A;
        for (size_t i = 0; i < numElementDOFs; ++i) {
            for (size_t j = 0; j < numElementDOFs; ++j) {
                A[i][j] = 0.0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    A[i][j] += w[k] * evalFunction(i, j, curl_b_q[k], val_b_q[k]);
                }
            }
        }
        IpplTimings::stopTimer(timerAxLocalMatrix);


        IpplTimings::TimerRef timerAxLoop = IpplTimings::getTimer("Ax Loop");
        IpplTimings::startTimer(timerAxLoop);

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex = elementIndices(index);
                
                // Here we now retrieve the global DOF indices and their
                // position inside of the FEMVector
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->NedelecSpace::getGlobalDOFIndices(elementIndex);
                
                const Vector<size_t, numElementDOFs> vectorIndices =
                    this->getFEMVectorDOFIndices(elementIndex, ldom);
                

                // local DOF indices
                size_t i, j;

                // global DOF n-dimensional indices (Vector of N indices
                // representing indices in each dimension)
                size_t I, J;
                
                for (i = 0; i < numElementDOFs; ++i) {
                    I = global_dofs[i];

                    // Skip boundary DOFs (Zero Dirichlet BCs)
                    if (this->isDOFOnBoundary(I)) {
                        continue;
                    }

                    for (j = 0; j < numElementDOFs; ++j) {
                        J = global_dofs[j];

                        // Skip boundary DOFs (Zero Dirichlet BCs)        
                        if (this->isDOFOnBoundary(J)) {
                            continue;
                        }

                        resultView(vectorIndices[i]) += A[i][j] * view(vectorIndices[j]);
                    }
                }
            }
        );
        IpplTimings::stopTimer(timerAxLoop);
        
        return resultVector;
    
    }


    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
                typename QuadratureType, typename FieldType>
    FEMVector<T> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::evaluateLoadVector(const FEMVector<NedelecSpace<T, Dim, Order, ElementType,
                                QuadratureType, FieldType>::point_t>& f) const {

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        const indices_t zeroNdIndex = Vector<size_t, Dim>(0);



        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }


        // Get the distance between the quadrature nodes and the DOFs 
        // we assume that the dofs are at the center of an edge, this is then
        // going to be used to implement a very crude interpolation scheme.
        Vector<Vector<T, numElementDOFs>, QuadratureType::numElementNodes>
            quadratureDOFDistances;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                point_t dofPos = getLocalDOFPosition(i);
                point_t d = dofPos - q[k];
                quadratureDOFDistances[k][i] = Kokkos::sqrt(d.dot(d));
            }
        }

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Get domain information and ghost cells
        auto ldom = layout_m.getLocalNDIndex();


        // Get boundary conditions from field
        FEMVector<T> resultVector = createFEMVector();

        // Get field data and make it atomic,
        // since it will be added to during the kokkos loop
        AtomicViewType atomic_view = resultVector.getView();
        typename detail::ViewType<point_t, 1>::view_type view = f.getView(); 

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(size_t index) {
                const size_t elementIndex                        = elementIndices(index);
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->NedelecSpace::getGlobalDOFIndices(elementIndex);

                const Vector<size_t, numElementDOFs> vectorIndices =
                    this->getFEMVectorDOFIndices(elementIndex, ldom);

                size_t i;

                for (i = 0; i < numElementDOFs; ++i) {
                    size_t I = global_dofs[i];
                    if (this->isDOFOnBoundary(I)) {
                        continue;
                    }
                        

                    // calculate the contribution of this element
                    T contrib = 0;
                    for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                        // We now have to interpolate the value of the field
                        // given at the DOF positions to the quadrature point.
                        point_t interpolatedVal(0);
                        T distSum = 0;
                        for (size_t j = 0; j < numElementDOFs; ++j) {
                            // get field index corresponding to this DOF

                            // the distance
                            T dist = quadratureDOFDistances[k][j];
                            
                            // running variable used for normalization
                            distSum += 1/dist;
                            
                            // get field value at DOF and interpolate to q_k
                            interpolatedVal += 1./dist * view(vectorIndices<:j:>);
                        }
                        // here we have to divide it by distSum in order to
                        // normalize it
                        interpolatedVal /= distSum;

                        // update contribution
                        contrib += w[k] * basis_q[k][i].dot(interpolatedVal) * absDetDPhi;
                    }

                    // add the contribution of the element to the field
                    atomic_view(vectorIndices<:i:>) += contrib;

                }
            });
        
        return resultVector;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
                typename QuadratureType, typename FieldType>
    template <typename F>
    FEMVector<T> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::evaluateLoadVectorFunctor(const F& f) const {

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        const indices_t zeroNdIndex = Vector<size_t, Dim>(0);

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }


        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Get domain information and ghost cells
        auto ldom = layout_m.getLocalNDIndex();


        // Get boundary conditions from field
        FEMVector<T> resultVector = createFEMVector();

        // Get field data and make it atomic,
        // since it will be added to during the kokkos loop
        AtomicViewType atomic_view = resultVector.getView();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;
        

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(size_t index) {
                const size_t elementIndex                        = elementIndices(index);
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->NedelecSpace::getGlobalDOFIndices(elementIndex);
                
                const Vector<size_t, numElementDOFs> vectorIndices =
                    this->getFEMVectorDOFIndices(elementIndex, ldom);
                

                size_t i, I;

                for (i = 0; i < numElementDOFs; ++i) {
                    I = global_dofs[i];

                    
                    if (this->isDOFOnBoundary(I)) {
                        continue;
                    }
                    
                    // calculate the contribution of this element
                    T contrib = 0;
                    for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                        // Get the global position of the quadrature point                        
                        point_t pos = this->ref_element_m.localToGlobal(
                            this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex)),
                            q[k]); 
                        
                        // evaluate the rhs function at this global position
                        point_t interpolatedVal = f(pos);
                        
                        // update contribution
                        contrib += w[k] * basis_q[k][i].dot(interpolatedVal) * absDetDPhi;
                    }

                    // add the contribution of the element to the vector
                    atomic_view(vectorIndices[i]) += contrib;
                
                }    
            });
        
        return resultVector;
    }



    ///////////////////////////////////////////////////////////////////////
    /// Basis functions and gradients /////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
                typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION typename NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>::point_t
                            NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::evaluateRefElementShapeFunction(const size_t& localDOF,
                                const NedelecSpace<T, Dim, Order, ElementType,
                                    QuadratureType, FieldType>::point_t& localPoint) const {

        // Assert that the local vertex index is valid.
        assert(localDOF < numElementDOFs && "The local vertex index is invalid"); 

        assert(this->ref_element_m.isPointInRefElement(localPoint)
            && "Point is not in reference element");
        


        // Simply hardcoded
        point_t result(0);
        if constexpr (Dim == 2) {
            T x = localPoint(0);
            T y = localPoint(1);

            switch (localDOF){
                case 0: result(0) = 1 - y; break;
                case 1: result(1) = 1 - x; break;
                case 2: result(0) = y; break;
                case 3: result(1) = x; break;
            }
        } else if constexpr (Dim == 3) {
            T x = localPoint(0);
            T y = localPoint(1);
            T z = localPoint(2);

            switch (localDOF){
                case 0:  result(0) = y*z - y - z + 1; break;
                case 1:  result(1) = x*z - x - z + 1; break;
                case 2:  result(0) = y*(1 - z);       break;
                case 3:  result(1) = x*(1 - z);       break;
                case 4:  result(2) = x*y - x - y + 1; break;
                case 5:  result(2) = x*(1 - y);       break;
                case 6:  result(2) = x*y;             break;
                case 7:  result(2) = y*(1 - x);       break;
                case 8:  result(0) = z*(1 - y);       break;
                case 9:  result(1) = z*(1 - x);       break;
                case 10: result(0) = y*z;             break;
                case 11: result(1) = x*z;             break;
            }
        }


        return result;
    }


    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION typename NedelecSpace<T, Dim, Order, ElementType,
                                QuadratureType, FieldType>::point_t
                             NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::evaluateRefElementShapeFunctionCurl(const size_t& localDOF,
                                    const NedelecSpace<T, Dim, Order, ElementType,
                                        QuadratureType, FieldType>::point_t& localPoint) const {
        
        // Hard coded.
        point_t result(0);

        if constexpr (Dim == 2) {
            // In case of 2d we would have that the curl would correspond to a
            // scalar, but in order to keep the interface uniform across all the
            // dimensions, we still use a 2d vector but only set the first entry
            // of it. This should lead to no problems, as we later on only will
            // take the dot product between two of them and therefore should not
            // run into any problems
            
            switch (localDOF) {
                case 0: result(0) = 1; break;
                case 1: result(0) = -1; break;
                case 2: result(0) = -1; break;
                case 3: result(0) = 1; break;
            }
        } else {
            T x = localPoint(0);
            T y = localPoint(1);
            T z = localPoint(2);

            switch (localDOF) {
                case 0: result(0) = 0; result(1) = -1+y; result(2) = 1-z; break;
                case 1: result(0) = 1-x; result(1) = 0; result(2) = -1+z; break;
                case 2: result(0) = 0; result(1) = -y; result(2) = -1+z; break;
                case 3: result(0) = x; result(1) = 0; result(2) = 1-z; break;
                case 4: result(0) = -1+x; result(1) = 1-y; result(2) = 0; break;
                case 5: result(0) = -x; result(1) = -1+y; result(2) = 0; break;
                case 6: result(0) = x; result(1) = -y; result(2) = 0; break;
                case 7: result(0) = 1-x; result(1) = y; result(2) = 0; break;
                case 8: result(0) = 0; result(1) = 1-y; result(2) = z; break;
                case 9: result(0) = -1+x; result(1) = 0; result(2) = -z; break;
                case 10: result(0) = 0; result(1) = y; result(2) = -z; break;
                case 11: result(0) = -x; result(1) = 0; result(2) = z; break;
            }
        }

        return result;
        
    }


    ///////////////////////////////////////////////////////////////////////
    /// FEMVector conversion //////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    FEMVector<T> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::createFEMVector() const {
        // This function will simply call one of the other two depending on the
        // dimension of the space
        if constexpr (Dim == 2) {
            return createFEMVector2d();
        } else {
            return createFEMVector3d();
        }
    }


    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    Kokkos::View<typename NedelecSpace<T, Dim, Order, ElementType, QuadratureType,
        FieldType>::point_t*>
    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>::reconstructToPoints(
        const Kokkos::View<typename NedelecSpace<T, Dim, Order, ElementType, QuadratureType,
        FieldType>::point_t*>& positions, const FEMVector<T>& coef) const {
        
        // The domain information of the subdomain of the MPI rank
        auto ldom = layout_m.getLocalNDIndex(); 
        
        // The domain information of the global domain
        auto gdom = layout_m.getDomain();
        indices_t gextent = gdom.last() - gdom.first();
        
        // The size of the global domain.
        point_t domainSize = (this->nr_m-1) * this->hr_m;


        auto coefView = coef.getView();

        Kokkos::View<point_t*> outView("reconstructed Func values at points", positions.extent(0));

        Kokkos::parallel_for("reconstructToPoints", positions.extent(0),
            KOKKOS_CLASS_LAMBDA(size_t i) {
                // get the current position and for it figure out to which
                // element it belongs
                point_t pos = positions<:i:>;
                indices_t elemIdx = ((pos - this->origin_m) / domainSize) * gextent;

            
                // next up we have to handle the case of when a position that
                // was provided to us lies on an edge at the upper bound of the
                // local domain, because in this case we have that the above
                // transformation gives back an element which is somewhat in the
                // halo. In order to fix this we simply subtract one.
                for (size_t d = 0; d < Dim; ++d) {
                    if (elemIdx<:d:> >= static_cast<size_t>(ldom.last()<:d:>)) {
                        elemIdx<:d:> -= 1;
                    }
                }


                // get correct indices
                const Vector<size_t, numElementDOFs> vectorIndices =
                    this->getFEMVectorDOFIndices(elemIdx, ldom);

                
                // figure out position inside of the reference element
                point_t locPos = pos - (elemIdx * this->hr_m + this->origin_m);
                locPos /= this->hr_m;

                // because of numerical instabilities it might happen then when
                // a point is on an edge this becomes marginally larger that 1 
                // or slightly negative which triggers an assertion. So this
                // simply is to prevent this.
                for (size_t d = 0; d < Dim; ++d) {
                    locPos<:d:> = Kokkos::min(T(1), locPos<:d:>);
                    locPos<:d:> = Kokkos::max(T(0), locPos<:d:>);
                }


                // interpolate the function value to the position, using the
                // basis functions.
                point_t val(0);
                for (size_t j = 0; j < numElementDOFs; ++j) {
                    point_t funcVal = this->evaluateRefElementShapeFunction(j, locPos);
                    val += funcVal*coefView(vectorIndices<:j:>);
                }
                outView(i) = val;

            }
        );
        

        return outView;
    }


    ///////////////////////////////////////////////////////////////////////
    /// Error norm computations ///////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    template <typename F>
    T NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>::computeError(
        const FEMVector<T>& u_h, const F& u_sol) const {
        if (this->quadrature_m.getOrder() < (2 * Order + 1)) {
            // throw exception
            throw IpplException( "NedelecSpace::computeError()",
                "Order of quadrature rule for error computation should be > 2*p + 1");
        }

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // Evaluate the basis functions for the DOF at the quadrature nodes
        Vector<Vector<point_t, numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        const indices_t zeroNdIndex = Vector<size_t, Dim>(0);

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m
            .getDeterminantOfTransformationJacobian(this->getElementMeshVertexPoints(zeroNdIndex)));

        // Variable to sum the error to
        T error = 0;

        // Get domain information and ghost cells
        auto ldom        = layout_m.getLocalNDIndex();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        auto view = u_h.getView();

    
        // Loop over elements to compute contributions
        Kokkos::parallel_reduce("Compute error over elements",
            policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(size_t index, double& local) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, numElementDOFs> global_dofs =
                    this->NedelecSpace::getGlobalDOFIndices(elementIndex);
                
                const Vector<size_t, numElementDOFs> vectorIndices =
                    this->getFEMVectorDOFIndices(elementIndex, ldom);


                // contribution of this element to the error
                T contrib = 0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    // Evaluate the analystical solution at the global position
                    // of the quadrature point
                    point_t val_u_sol = u_sol(this->ref_element_m.localToGlobal(
                        this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex)),
                            q[k]));
                    
                    // Here we now reconstruct the solution given the basis
                    // functions.
                    point_t val_u_h = 0;
                    for (size_t j = 0; j < numElementDOFs; ++j) {
                        // get field index corresponding to this DOF
                        val_u_h += basis_q[k][j] * view(vectorIndices[j]);
                    }

                    // calculate error and add to sum.
                    point_t dif = (val_u_sol -  val_u_h);
                    T x = dif.dot(dif);
                    contrib += w[k] * x * absDetDPhi;
                }
                local += contrib;
            },
            Kokkos::Sum<double>(error)
        );

        // MPI reduce
        T global_error = 0.0;
        Comm->allreduce(error, global_error, 1, std::plus<T>());

        return Kokkos::sqrt(global_error);
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION bool NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::isDOFOnBoundary(const size_t& dofIdx) const {
        
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
                // space
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



    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION int NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::getBoundarySide(const size_t& dofIdx) const {

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
                // space
                size_t f = dofIdx - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset
                    - (nx*(ny-1) + ny*(nx-1));
                
                size_t yOffset = f / nx;
                // South
                if (yOffset == 0) return 0;
                // North
                if (yOffset == ny-1) return 2;

                size_t xOffset = f % nx;
                // West
                if (xOffset == 0) return 1;
                // East
                if (xOffset == nx-1) return 3;

            } else {
                // are parallel to one of the other axes
                // Ground
                if (zOffset == 0) return 4;
                // Space
                if (zOffset == nz-1) return 5;
                
                size_t f = dofIdx - (nx*(ny-1) + ny*(nx-1) + nx*ny)*zOffset;
                size_t yOffset = f / (2*nx - 1);
                size_t xOffset = f - (2*nx - 1)*yOffset;

                if (xOffset < (nx-1)) {
                    // we are parallel to the x axis, therefore we cannot
                    // be on an west or east boundary, but we still can
                    // be on a north or south boundary
                    
                    // South
                    if (yOffset == 0) return 0;
                    // North
                    if (yOffset == ny-1) return 2;
                    
                } else {
                    // we are parallel to the y axis, therefore we cannot be
                    // on a south or north boundary, but we still can be on
                    // a west or east boundary
                    if (xOffset >= nx-1) {
                        xOffset -= (nx-1);
                    }

                    // West
                    if (xOffset == 0) return 1;
                    // East
                    if (xOffset == nx-1) return 3;
                }
            }
            return -1;
        }

    }


    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    FEMVector<T> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::createFEMVector2d() const{

        // Here we will create an empty FEMVector for the case of the domain
        // being 2D.
        // The largest part of this is going to be the handling of the halo
        // cells, more specifically figuring out which entries of the vector are
        // part of the sendIdxs and which are part of the recvIdxs. For this we
        // loop through all the other domains and for each of them figure out if
        // we share a boundary with them. If we share a boundary we then have to
        // figure out which entries of the vector are part of this boundary. To
        // do this we loop though all the mesh elements which are on this
        // boundary and then for each of these elements we take the DOFs which
        // should be part of the sendIdxs and the ones which are part of the
        // recvIdxs. Currently in this step we have to manually select the
        // correct DOFs of the reference element corresponding to that specific
        // boundary type (north, south, west, east). This manual selection of
        // the correct DOFs might lead to an "out of the blue" feeling on why
        // exactly we selected those DOFs, but it should be easily verifyable
        // that those are the correct DOFs.
        // Also note that we are not exchanging any boundary information over 
        // corners, test showed that this does not have any impact on the 
        // correctness.
        // For more information regarding the domain decomposition refer to the
        // report available at: TODO add reference to report on AMAS website

        auto ldom = layout_m.getLocalNDIndex();
        auto doms = layout_m.getHostLocalDomains();

        // Create the temporaries and so on which will store the MPI
        // information.
        std::vector<size_t> neighbors;
        std::vector< Kokkos::View<size_t*> > sendIdxs;
        std::vector< Kokkos::View<size_t*> > recvIdxs;
        std::vector< std::vector<size_t> > sendIdxsTemp;
        std::vector< std::vector<size_t> > recvIdxsTemp;

        // Here we loop thought all the domains to figure out how we are related
        // to them and if we have to do any kind of exchange.
        size_t myRank = Comm->rank();
        for (size_t i = 0; i < doms.extent(0); ++i) {
            if (i == myRank) {
                // We are looking at ourself
                continue;
            }
            auto odom = doms(i);

            // East boundary
            if (ldom.last()[0] == odom.first()[0]-1 &&
                    !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
                // Extract the range of the boundary.
                int begin = std::max(odom.first()[1], ldom.first()[1]);
                int end = std::min(odom.last()[1], ldom.last()[1]);
                int pos = ldom.last()[0];
                
                // Add this to the neighbour list.
                neighbors.push_back(i);
                sendIdxsTemp.push_back(std::vector<size_t>());
                recvIdxsTemp.push_back(std::vector<size_t>());
                size_t idx = neighbors.size() - 1;
                
                // Add all the halo
                indices_t elementPosHalo(0);
                elementPosHalo(0) = pos;
                indices_t elementPosSend(0);
                elementPosSend(0) = pos;
                for (int k = begin; k <= end; ++k) {
                    elementPosHalo(1) = k;
                    elementPosSend(1) = k;
                    
                    auto dofIndicesHalo = getFEMVectorDOFIndices(elementPosHalo, ldom);
                    recvIdxsTemp[idx].push_back(dofIndicesHalo[3]);

                    auto dofIndicesSend = getFEMVectorDOFIndices(elementPosSend, ldom);
                    sendIdxsTemp[idx].push_back(dofIndicesSend[0]);
                    sendIdxsTemp[idx].push_back(dofIndicesSend[1]);
                }
                // Check if on very north
                if (end == layout_m.getDomain().last()[1] || ldom.last()[1] > odom.last()[1]) {
                    elementPosSend(1) = end;
                    auto dofIndicesSend = getFEMVectorDOFIndices(elementPosSend, ldom);
                    // also have to add dof 2
                    sendIdxsTemp[idx].push_back(dofIndicesSend[2]);
                }
            }

            // West boundary
            if (ldom.first()[0] == odom.last()[0]+1 &&
                    !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
                // Extract the range of the boundary.
                int begin = std::max(odom.first()[1], ldom.first()[1]);
                int end = std::min(odom.last()[1], ldom.last()[1]);
                int pos = ldom.first()[0];
                
                // Add this to the neighbour list.
                neighbors.push_back(i);
                sendIdxsTemp.push_back(std::vector<size_t>());
                recvIdxsTemp.push_back(std::vector<size_t>());
                size_t idx = neighbors.size() - 1;
                
                // Add all the halo
                indices_t elementPosHalo(0);
                elementPosHalo(0) = pos-1;
                indices_t elementPosSend(0);
                elementPosSend(0) = pos;
                for (int k = begin; k <= end; ++k) {
                    elementPosHalo(1) = k;
                    elementPosSend(1) = k;
                    
                    auto dofIndicesHalo = getFEMVectorDOFIndices(elementPosHalo, ldom);
                    recvIdxsTemp[idx].push_back(dofIndicesHalo[0]);
                    recvIdxsTemp[idx].push_back(dofIndicesHalo[1]);

                    auto dofIndicesSend = getFEMVectorDOFIndices(elementPosSend, ldom);
                    sendIdxsTemp[idx].push_back(dofIndicesSend[1]);
                }
                // Check if on very north
                if (end == layout_m.getDomain().last()[1] || odom.last()[1] > ldom.last()[1]) {
                    elementPosHalo(1) = end;
                    auto dofIndicesHalo = getFEMVectorDOFIndices(elementPosHalo, ldom);
                    // also have to add dof 2
                    recvIdxsTemp[idx].push_back(dofIndicesHalo[2]);
                }
            }

            // North boundary
            if (ldom.last()[1] == odom.first()[1]-1 &&
                    !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
                // Extract the range of the boundary.
                int begin = std::max(odom.first()[0], ldom.first()[0]);
                int end = std::min(odom.last()[0], ldom.last()[0]);
                int pos = ldom.last()[1];
                
                // Add this to the neighbour list.
                neighbors.push_back(i);
                sendIdxsTemp.push_back(std::vector<size_t>());
                recvIdxsTemp.push_back(std::vector<size_t>());
                size_t idx = neighbors.size() - 1;
                
                // Add all the halo
                indices_t elementPosHalo(0);
                elementPosHalo(1) = pos;
                indices_t elementPosSend(0);
                elementPosSend(1) = pos;
                for (int k = begin; k <= end; ++k) {
                    elementPosHalo(0) = k;
                    elementPosSend(0) = k;
                    
                    auto dofIndicesHalo = getFEMVectorDOFIndices(elementPosHalo, ldom);
                    recvIdxsTemp[idx].push_back(dofIndicesHalo[2]);

                    auto dofIndicesSend = getFEMVectorDOFIndices(elementPosSend, ldom);
                    sendIdxsTemp[idx].push_back(dofIndicesSend[0]);
                    sendIdxsTemp[idx].push_back(dofIndicesSend[1]);
                }
                // Check if on very east
                if (end == layout_m.getDomain().last()[0] || ldom.last()[0] > odom.last()[0]) {
                    elementPosSend(0) = end;
                    auto dofIndicesSend = getFEMVectorDOFIndices(elementPosSend, ldom);
                    // also have to add dof 3
                    sendIdxsTemp[idx].push_back(dofIndicesSend[3]);
                }
            }

            // South boundary
            if (ldom.first()[1] == odom.last()[1]+1 &&
                    !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
                // Extract the range of the boundary.
                int begin = std::max(odom.first()[0], ldom.first()[0]);
                int end = std::min(odom.last()[0], ldom.last()[0]);
                int pos = ldom.first()[1];
                
                // Add this to the neighbour list.
                neighbors.push_back(i);
                sendIdxsTemp.push_back(std::vector<size_t>());
                recvIdxsTemp.push_back(std::vector<size_t>());
                size_t idx = neighbors.size() - 1;
                
                // Add all the halo
                indices_t elementPosHalo(0);
                elementPosHalo(1) = pos-1;
                indices_t elementPosSend(0);
                elementPosSend(1) = pos;
                for (int k = begin; k <= end; ++k) {
                    elementPosHalo(0) = k;
                    elementPosSend(0) = k;
                    
                    auto dofIndicesHalo = getFEMVectorDOFIndices(elementPosHalo, ldom);
                    recvIdxsTemp[idx].push_back(dofIndicesHalo[0]);
                    recvIdxsTemp[idx].push_back(dofIndicesHalo[1]);

                    auto dofIndicesSend = getFEMVectorDOFIndices(elementPosSend, ldom);
                    sendIdxsTemp[idx].push_back(dofIndicesSend[0]);
                }
                // Check if on very east
                if (end == layout_m.getDomain().last()[0] || odom.last()[0] > ldom.last()[0]) {
                    elementPosHalo(0) = end;
                    auto dofIndicesHalo = getFEMVectorDOFIndices(elementPosHalo, ldom);
                    // also have to add dof 3
                    recvIdxsTemp[idx].push_back(dofIndicesHalo[3]);
                }
            }
        }



        // Here we now have to translate the sendIdxsTemp and recvIdxsTemp which
        // are std::vectors<std::vector> into the correct list type which
        // is std::vector<Kokkos::View>
        for (size_t i = 0; i < neighbors.size(); ++i) {
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
        indices_t extents(0);
        extents = (ldom.last() - ldom.first()) + 3;
        size_t nx = extents(0);
        size_t ny = extents(1);
        size_t n = nx*(ny-1) + ny*(nx-1);
        FEMVector<T> vec(n, neighbors, sendIdxs, recvIdxs);
        
        return vec;
    }



    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    FEMVector<T> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::createFEMVector3d() const{



        
        // Here we will create an empty FEMVector for the case of the domain
        // being 3D.
        // It follows the same principle as the 2D case (check comment there).
        // The major difference now is that we have more types of boundaries.
        // Namely we have 6 directions: west, east, south, north, ground, and
        // space. Where west-east is on the x-axis, south-north on the y-axis,
        // and ground-space on the z-axis. For this we have 3 major types of
        // boundaries, namely "flat" boundaries which are along the coordinate
        // axes (your standard west-east, south-north, and ground-space
        // exchanges), then we have two different types of diagonal exchanges,
        // which we will call "positive" and "negative", we have two types as
        // a diagonal exchange always happens over an edge and this edges is 
        // shared by 4 different ranks, so we have two different diagonal 
        // exchanges per edge, which we differentiate with "positive" and 
        // "negative".
        // If we now look at these types independently we have that the code
        // needed to perform one such exchange is large going to be independent
        // of the direction of the exchange (i.e. is the "flat" exchange
        // happening over a west-est or a ground-space boundary) the major
        // difference is the DOF indices we have to chose for the elements on
        // the boundary (check the 2D case for more info). 
        // We therefore create 3 lambdas for these different types of boundaries
        // which we then call with appropriate arguments for the direction of the
        // exchange.
        // Note that like with the 2D case we do not consider any exchanges over
        // corners.
        // For more information regarding the domain decomposition refer to the
        // report available at: TODO add reference to report on AMAS website

        using indices_t = Vector<int, Dim>;

        auto ldom = layout_m.getLocalNDIndex();
        auto doms = layout_m.getHostLocalDomains();

        // Create the temporaries and so on which will store the MPI
        // information.
        std::vector<size_t> neighbors;
        std::vector< Kokkos::View<size_t*> > sendIdxs;
        std::vector< Kokkos::View<size_t*> > recvIdxs;
        std::vector< std::vector<size_t> > sendIdxsTemp;
        std::vector< std::vector<size_t> > recvIdxsTemp;

        // The parameters are:
        // i: The index of the other dom we are looking at (according to doms).
        // a: Along which axis the exchange happens, 0 = x-axis, 1 = y-axis,
        //      2 = z-axis.
        // f,s: While the exchange happens over the axis "a" we have that
        //        elements which are part of the boundary are on a plane spanned
        //        by the other two axes, these other two axes are then given by
        //        these two variables "f" and "s" (they then also define the
        //        order in which we go though these axes).
        // va,vb: These are the placeholders for the sendIdxs and recvIdxs
        //          arrays, note that depending on if an exchange happens from,
        //          e.g. west to east or from east to west the role of which of 
        //          these placeholders stores the sendIdxs and which one stores
        //          the recvIdxs changes.
        // posA, posB: The "a"-axis coordinate of the elements which are part of
        //               the boundary, we have two of them as the coordinate
        //               can be different depending on if we are looking at the
        //               sendIdxs or the recvIdxs.
        // idxsA, idxsB: These are going to be the local DOF indices of the
        //                 elements which are part of the boundary and which
        //                 need to be exchanged. Again we have two as the
        //                 indices are going to depend on if we are looking at
        //                 the sendIdxs or the recvIdxs
        // adom, bdom: The domain extents of the two domains which are part of
        //               this exchange.
        auto flatBoundaryExchange = [this, &neighbors, &ldom](
            size_t i, size_t a, size_t f, size_t s,
            std::vector<std::vector<size_t> >& va, std::vector<std::vector<size_t> >& vb,
            int posA, int posB,
            const std::vector<size_t>& idxsA, const std::vector<size_t>& idxsB,
            NDIndex<3>& adom, NDIndex<3>& bdom) {
            
            int beginF = std::max(bdom.first()[f], adom.first()[f]);
            int endF = std::min(bdom.last()[f], adom.last()[f]);
            int beginS = std::max(bdom.first()[s], adom.first()[s]);
            int endS = std::min(bdom.last()[s], adom.last()[s]);
            
            neighbors.push_back(i);
            va.push_back(std::vector<size_t>());
            vb.push_back(std::vector<size_t>());
            size_t idx = neighbors.size() - 1;


            indices_t elementPosA(0);
            elementPosA(a) = posA;
            indices_t elementPosB(0);
            elementPosB(a) = posB;

            // Here we now have the double loop that goes though all the
            // elements spanned by the plane given by the "f" and "s" axis.
            for (int k = beginF; k <= endF; ++k) {
                elementPosA(f) = k;
                elementPosB(f) = k;
                for (int l = beginS; l <= endS; ++l) {
                    elementPosA(s) = l;
                    elementPosB(s) = l;

                    auto dofIndicesA = this->getFEMVectorDOFIndices(elementPosA, ldom);
                    va[idx].push_back(dofIndicesA[idxsA[0]]);
                    va[idx].push_back(dofIndicesA[idxsA[1]]);

                    auto dofIndicesB = this->getFEMVectorDOFIndices(elementPosB, ldom);
                    vb[idx].push_back(dofIndicesB[idxsB[0]]);
                    vb[idx].push_back(dofIndicesB[idxsB[1]]);
                    vb[idx].push_back(dofIndicesB[idxsB[2]]);

                    
                    // We now have reached the end of the first axis and have to
                    // figure out if we need to add any additional DOFs. If we 
                    // need to add DOFs depends on if we are at the mesh
                    // boundary and if one of the two domains does not end here
                    // and "overlaps" the other one.
                    if (k == endF) {
                        if (endF == layout_m.getDomain().last()[f] ||
                                bdom.last()[f] > adom.last()[f]) {
                            va[idx].push_back(dofIndicesA[idxsA[2]]);
                        }
                        
                        if (endF == layout_m.getDomain().last()[f] ||
                                adom.last()[f] > bdom.last()[f]) {
                            vb[idx].push_back(dofIndicesB[idxsB[3]]);
                            vb[idx].push_back(dofIndicesB[idxsB[4]]);
                        }
                        
                        // This is a modification to the beginning of the f axis
                        // we still put it on here as we are guaranteed that
                        // like this it only gets called once
                        // call this last, as modifies elementPosA(s) 
                        if (bdom.first()[f] < adom.first()[f]) {
                            indices_t tmpPos = elementPosA;
                            tmpPos(f) = beginF-1;
                            auto dofIndicestmp = this->getFEMVectorDOFIndices(tmpPos, ldom);
                            va[idx].push_back(dofIndicestmp[idxsA[0]]);
                            va[idx].push_back(dofIndicestmp[idxsA[1]]);
                        }
                    }
                }
                
                // We now have reached the end of the second axis and have to
                // figure out if we need to add any additional DOFs. If we need
                // to add DOFs depends on if we are at the mesh boundary and if
                // one of the two domains does not end here and "overlaps" the
                // other one.

                if (endS == layout_m.getDomain().last()[s] || bdom.last()[s] > adom.last()[s]) {
                    elementPosA(s) = endS;
                    auto dofIndicesA = this->getFEMVectorDOFIndices(elementPosA, ldom);
                    va[idx].push_back(dofIndicesA[idxsA[3]]);
                }
                
                if (endS == layout_m.getDomain().last()[s] || adom.last()[s] > bdom.last()[s]) {
                    elementPosB(s) = endS;
                    auto dofIndicesB = this->getFEMVectorDOFIndices(elementPosB, ldom);
                    vb[idx].push_back(dofIndicesB[idxsB[5]]);
                    vb[idx].push_back(dofIndicesB[idxsB[6]]);
                }
                
                // This is a modification to the beginning of the s axis
                // we still put it on here as we are guaranteed that
                // like this it only gets called once
                // call this last, as modifies elementPosA(f);
                if (bdom.first()[f] < adom.first()[f]) {
                    indices_t tmpPos = elementPosA;
                    tmpPos(s) = beginS-1;
                    auto dofIndicestmp = this->getFEMVectorDOFIndices(tmpPos, ldom);
                    va[idx].push_back(dofIndicestmp[idxsA[0]]);
                    va[idx].push_back(dofIndicestmp[idxsA[1]]);
                }
            }
            // At this point we have reached the end of both axes f and s and 
            // therefore now have to make one final check.
            if ((endF == layout_m.getDomain().last()[f] || adom.last()[f] > bdom.last()[f]) && 
                (endS == layout_m.getDomain().last()[s] || adom.last()[s] > bdom.last()[s])) {
                elementPosB(f) = endF;
                elementPosB(s) = endS;
                auto dofIndicesB = this->getFEMVectorDOFIndices(elementPosB, ldom);
                vb[idx].push_back(dofIndicesB[idxsB[7]]);
            }
        };

        // The parameters are:
        // i: The index of the other dom we are looking at (according to doms).
        // a: Along which axis the edge is over which the exchange happens,
        //      0 = x-axis, 1 = y-axis, 2 = z-axis.
        // f,s: While the exchange happens over the edge along the  axis "a" we
        //        have store in those two the other two axes.
        // ao,bo: These are offset variables as certain exchanges require
        //          offsets to certain values.
        // va,vb: These are the placeholders for the sendIdxs and recvIdxs
        //          arrays, note that depending on if an exchange happens from,
        //          e.g. west to east or from east to west the role of which of 
        //          these placeholders stores the sendIdxs and which one stores
        //          the recvIdxs changes.
        // idxsA, idxsB: These are going to be the local DOF indices of the
        //                 elements which are part of the boundary and which
        //                 need to be exchanged. Again we have two as the
        //                 indices are going to depend on if we are looking at
        //                 the sendIdxs or the recvIdxs
        // odom: The other domain we are exchanging to.
        auto negativeDiagonalExchange = [this, &neighbors, &ldom](
            size_t i, size_t a, size_t f, size_t s, int ao, int bo,
            std::vector<std::vector<size_t> >& va, std::vector<std::vector<size_t> >& vb,
            const std::vector<size_t>& idxsA, const std::vector<size_t>& idxsB,
            NDIndex<3>& odom) {
            
            neighbors.push_back(i);
            va.push_back(std::vector<size_t>());
            vb.push_back(std::vector<size_t>());
            size_t idx = neighbors.size() - 1;

            indices_t elementPosA(0);
            elementPosA(f) = ldom.last()[f];
            elementPosA(s) = ldom.first()[s] + ao;

            indices_t elementPosB(0);
            elementPosB(f) = ldom.last()[f];
            elementPosB(s) = ldom.first()[s] + bo;

            int begin = std::max(odom.first()[a], ldom.first()[a]);
            int end = std::min(odom.last()[a], ldom.last()[a]);
            // Loop through all the elements along the edge.
            for (int k = begin; k <= end; ++k) {
                elementPosA(a) = k;
                elementPosB(a) = k;

                auto dofIndicesA = this->getFEMVectorDOFIndices(elementPosA, ldom);
                va[idx].push_back(dofIndicesA[idxsA[0]]);
                va[idx].push_back(dofIndicesA[idxsA[1]]);

                auto dofIndicesB = this->getFEMVectorDOFIndices(elementPosB, ldom);
                vb[idx].push_back(dofIndicesB[idxsB[0]]);
                vb[idx].push_back(dofIndicesB[idxsB[1]]);
            }
        };


        // The parameters are:
        // i: The index of the other dom we are looking at (according to doms).
        // a: Along which axis the edge is over which the exchange happens,
        //      0 = x-axis, 1 = y-axis, 2 = z-axis.
        // f,s: While the exchange happens over the edge along the  axis "a" we
        //        have store in those two the other two axes.
        // va,vb: These are the placeholders for the sendIdxs and recvIdxs
        //          arrays, note that depending on if an exchange happens from,
        //          e.g. west to east or from east to west the role of which of 
        //          these placeholders stores the sendIdxs and which one stores
        //          the recvIdxs changes.
        // idxsA, idxsB: These are going to be the local DOF indices of the
        //                 elements which are part of the boundary and which
        //                 need to be exchanged. Again we have two as the
        //                 indices are going to depend on if we are looking at
        //                 the sendIdxs or the recvIdxs
        // odom: The other domain we are exchanging to.
        auto positiveDiagonalExchange = [this, &neighbors, &ldom](
            size_t i, size_t a, size_t f, size_t s,
            indices_t posA, indices_t posB,
            std::vector<std::vector<size_t> >& va, std::vector<std::vector<size_t> >& vb,
            const std::vector<size_t>& idxsA, const std::vector<size_t>& idxsB,
            NDIndex<3>& odom) {
            
            neighbors.push_back(i);
            va.push_back(std::vector<size_t>());
            vb.push_back(std::vector<size_t>());
            size_t idx = neighbors.size() - 1;

            indices_t elementPosA(0);
            elementPosA(f) = posA(f);
            elementPosA(s) = posA(s);

            indices_t elementPosB(0);
            elementPosB(f) = posB(f);
            elementPosB(s) = posB(s);

            int begin = std::max(odom.first()[a], ldom.first()[a]);
            int end = std::min(odom.last()[a], ldom.last()[a]);

            for (int k = begin; k <= end; ++k) {
                elementPosA(a) = k;
                elementPosB(a) = k;

                auto dofIndicesA = this->getFEMVectorDOFIndices(elementPosA, ldom);
                va[idx].push_back(dofIndicesA[idxsA[0]]);
                va[idx].push_back(dofIndicesA[idxsA[1]]);
                va[idx].push_back(dofIndicesA[idxsA[2]]);

                auto dofIndicesB = this->getFEMVectorDOFIndices(elementPosB, ldom);
                vb[idx].push_back(dofIndicesB[idxsB[0]]);
            }
        };

        // After we now have defined the code required for each of the exchanges
        // we can look at all the exchanges we need to make and call the lambdas
        // with the appropriate parameters.


        // Here we loop through all the domains to figure out how we are related
        // to them and if we have to do any kind of exchange.
        size_t myRank = Comm->rank();
        for (size_t i = 0; i < doms.extent(0); ++i) {
            if (i == myRank) {
                // We are looking at ourself
                continue;
            }
            auto odom = doms(i);

            // East boundary
            if (ldom.last()[0] == odom.first()[0]-1 &&
                    !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1]) && 
                    !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
                
                int pos = ldom.last()[0];
                flatBoundaryExchange(
                    i, 0, 1, 2,
                    recvIdxsTemp, sendIdxsTemp,
                    pos, pos,
                    {3,5,6,11}, {0,1,4,2,7,8,9,10},
                    ldom, odom
                );
            }

            // West boundary
            if (ldom.first()[0] == odom.last()[0]+1 &&
                    !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1]) && 
                    !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
                
                int pos = ldom.first()[0];
                flatBoundaryExchange(
                    i, 0, 1, 2,
                    sendIdxsTemp, recvIdxsTemp,
                    pos, pos-1,
                    {1,4,7,9}, {0,1,4,2,7,8,9,10},
                    odom, ldom
                );
            }

            // North boundary
            if (ldom.last()[1] == odom.first()[1]-1 &&
                    !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0]) &&
                    !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {

                int pos = ldom.last()[1];
                flatBoundaryExchange(
                    i, 1, 0, 2,
                    recvIdxsTemp, sendIdxsTemp,
                    pos, pos,
                    {2,7,6,10}, {0,1,4,3,5,8,9,11},
                    ldom, odom
                );
            }

            // South boundary
            if (ldom.first()[1] == odom.last()[1]+1 &&
                    !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0]) &&
                    !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
                
                int pos = ldom.first()[1];
                flatBoundaryExchange(
                    i, 1, 0, 2,
                    sendIdxsTemp, recvIdxsTemp,
                    pos, pos-1,
                    {0,4,5,8}, {0,1,4,3,5,8,9,11},
                    odom, ldom
                );
                
            }

            // Space boundary
            if (ldom.last()[2] == odom.first()[2]-1 &&
                    !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0]) && 
                    !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
                
                int pos = ldom.last()[2];
                flatBoundaryExchange(
                    i, 2, 0, 1,
                    recvIdxsTemp, sendIdxsTemp,
                    pos, pos,
                    {8,9,11,10}, {0,1,4,3,5,2,7,6},
                    ldom, odom
                );
            }

            // Ground boundary
            if (ldom.first()[2] == odom.last()[2]+1 &&
                    !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0]) && 
                    !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {

                int pos = ldom.first()[2];
                flatBoundaryExchange(
                    i, 2, 0, 1,
                    sendIdxsTemp, recvIdxsTemp,
                    pos, pos-1,
                    {0,1,3,2}, {0,1,4,3,5,2,7,6},
                    odom, ldom
                );
            }


            
            // Next up we handle all the annoying diagonals.
            // The negative ones:
            // Parallel to y from space to ground, west to east
            if (ldom.last()[0] == odom.first()[0]-1 && ldom.first()[2] == odom.last()[2]+1 && 
                    !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
                
                negativeDiagonalExchange(
                    i, 1, 0, 2, 0, -1,
                    sendIdxsTemp, recvIdxsTemp,
                    {0,1}, {3,5},
                    odom
                );
            }

            // Parallel to y from ground to space, east to west
            if (ldom.first()[0] == odom.last()[0]+1 && ldom.last()[2] == odom.first()[2]-1 && 
                    !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
                
                negativeDiagonalExchange(
                    i, 1, 2, 0, -1, 0,
                    recvIdxsTemp, sendIdxsTemp,
                    {8,9}, {1,4},
                    odom
                );
            }


            // Parallel to x from space to ground, south to north
            if (ldom.last()[1] == odom.first()[1]-1 && ldom.first()[2] == odom.last()[2]+1 && 
                    !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
                negativeDiagonalExchange(
                    i, 0, 1, 2, 0, -1,
                    sendIdxsTemp, recvIdxsTemp,
                    {0,1}, {2,7},
                    odom
                );
            }

            // Parallel to x from ground to space, north to south
            if (ldom.first()[1] == odom.last()[1]+1 && ldom.last()[2] == odom.first()[2]-1 && 
                    !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
                negativeDiagonalExchange(
                    i, 0, 2, 1, -1, 0,
                    recvIdxsTemp, sendIdxsTemp,
                    {8,9}, {0,4},
                    odom
                );
            }


            // Parallel to z from west to east, north to south
            if (ldom.last()[0] == odom.first()[0]-1 && ldom.first()[1] == odom.last()[1]+1 && 
                    !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
                negativeDiagonalExchange(
                    i, 2, 0, 1, 0, -1,
                    sendIdxsTemp, recvIdxsTemp,
                    {0,4}, {3,5},
                    odom
                );
            }

            // Parallel to z from east to west, south to north
            if (ldom.first()[0] == odom.last()[0]+1 && ldom.last()[1] == odom.first()[1]-1 && 
                    !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
                negativeDiagonalExchange(
                    i, 2, 1, 0, -1, 0,
                    recvIdxsTemp, sendIdxsTemp,
                    {2,7}, {1,4},
                    odom
                );
            }



            // The positive ones
            // Parallel to y from ground to space, west to east
            if (ldom.last()[0] == odom.first()[0]-1 && ldom.last()[2] == odom.first()[2]-1 && 
                    !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
                positiveDiagonalExchange(
                    i, 1, 0, 2,
                    ldom.last(), ldom.last(),
                    sendIdxsTemp, recvIdxsTemp,
                    {0,1,4}, {11},
                    odom
                );
            }

            // Parallel to y from space to ground, east to west
            if (ldom.first()[0] == odom.last()[0]+1 && ldom.first()[2] == odom.last()[2]+1 && 
                    !(odom.last()[1] < ldom.first()[1] || odom.first()[1] > ldom.last()[1])) {
                positiveDiagonalExchange(
                    i, 1, 0, 2,
                    ldom.first()-1, ldom.first(),
                    recvIdxsTemp, sendIdxsTemp,
                    {0,1,4}, {1},
                    odom
                );
            }


            // Parallel to x from ground to space, south to north
            if (ldom.last()[1] == odom.first()[1]-1 && ldom.last()[2] == odom.first()[2]-1 && 
                    !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
                positiveDiagonalExchange(
                    i, 0, 1, 2,
                    ldom.last(), ldom.last(),
                    sendIdxsTemp, recvIdxsTemp,
                    {0,1,4}, {10},
                    odom
                );
            }

            // Parallel to x from space to ground, north to south
            if (ldom.first()[1] == odom.last()[1]+1 && ldom.first()[2] == odom.last()[2]+1 && 
                    !(odom.last()[0] < ldom.first()[0] || odom.first()[0] > ldom.last()[0])) {
                positiveDiagonalExchange(
                    i, 0, 1, 2,
                    ldom.first()-1, ldom.first(),
                    recvIdxsTemp, sendIdxsTemp,
                    {0,1,4}, {0},
                    odom
                );
            }


            // Parallel to z from west to east, south to north
            if (ldom.last()[0] == odom.first()[0]-1 && ldom.last()[1] == odom.first()[1]-1 && 
                    !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
                positiveDiagonalExchange(
                    i, 2, 0, 1,
                    ldom.last(), ldom.last(),
                    sendIdxsTemp, recvIdxsTemp,
                    {0,1,4}, {6},
                    odom
                );
            }

            // Parallel to z from east to west, north to south
            if (ldom.first()[0] == odom.last()[0]+1 && ldom.first()[1] == odom.last()[1]+1 && 
                    !(odom.last()[2] < ldom.first()[2] || odom.first()[2] > ldom.last()[2])) {
                positiveDiagonalExchange(
                    i, 2, 0, 1,
                    ldom.first()-1, ldom.first(),
                    recvIdxsTemp, sendIdxsTemp,
                    {0,1,4}, {4},
                    odom
                );
            }
            
        }
        



        // Here we now have to translate the sendIdxsTemp and recvIdxsTemp which
        // are std::vectors<std::vector> into the correct list type which
        // is std::vector<Kokkos::View>
        for (size_t i = 0; i < neighbors.size(); ++i) {
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
        indices_t extents(0);
        extents = (ldom.last() - ldom.first()) + 3;
        size_t nx = extents(0);
        size_t ny = extents(1);
        size_t nz = extents(2);
        size_t n = (nz-1)*(nx*(ny-1) + ny*(nx-1) + nx*ny) + nx*(ny-1) + ny*(nx-1);
        FEMVector<T> vec(n, neighbors, sendIdxs, recvIdxs);
        
        return vec;
    }



}  // namespace ippl
