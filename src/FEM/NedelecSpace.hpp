
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
    // Sets the correct mesh ad decomposes the elements among ranks according to layout.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    void NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::initialize(UniformCartesian<T, Dim>& mesh, const Layout_t& layout) {
        
        FiniteElementSpace<T, Dim, getNedelecNumElementDOFs(Dim, Order),
                            ElementType, QuadratureType, FEMVector<T>, FEMVector<T>>::setMesh(mesh);

        // Initialize the elementIndices view
        initializeElementIndices(layout);
    }

    // Initialize element indices Kokkos View
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    void NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::initializeElementIndices(const Layout_t& layout) {
        
        size_t nx = this->nr_m[0];
        size_t ny = this->nr_m[1];
        size_t n = (nx-1) * (ny-1);
        elementIndices = Kokkos::View<size_t*>("elementIndices", n);
        Kokkos::parallel_for("ComputeElementIndices",n,
            KOKKOS_CLASS_LAMBDA(size_t i){
                elementIndices(i) = i;
            }
        );
        /*
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
        Kokkos::parallel_reduce("ComputePoints", npoints,
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
            Kokkos::Sum<int>(upperBoundaryPoints)
        );

        Kokkos::fence();

        int elementsPerRank = npoints - upperBoundaryPoints;
        elementIndices      = Kokkos::View<size_t*>("i", elementsPerRank);
        Kokkos::View<size_t> index("index");

        Kokkos::parallel_for("RemoveNaNs", npoints, KOKKOS_CLASS_LAMBDA(const int i) {
                if ((points(i) != 0) || (i == 0)) {
                    const size_t idx    = Kokkos::atomic_fetch_add(&index(), 1);
                    elementIndices(idx) = points(i);
                }
            });
        */
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
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION size_t NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::getGlobalDOFIndex(const size_t& elementIndex,
                                const size_t& localDOFIndex) const {

        const auto global_dofs = this->getGlobalDOFIndices(elementIndex);

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
                            ::getGlobalDOFIndices(const size_t& elementIndex) const {

        Vector<size_t, this->numElementDOFs> globalDOFs(0);
        indices_t elementPos = this->getElementNDIndex(elementIndex);

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

        size_t nx = this->nr_m[0];
        globalDOFs(0) = v.dot(elementPos);
        globalDOFs(1) = globalDOFs(0) + nx - 1;
        globalDOFs(2) = globalDOFs(1) + nx;
        globalDOFs(3) = globalDOFs(1) + 1;

        if constexpr (Dim == 3) {
            size_t ny = this->nr_m[1];

            globalDOFs(4) = v(2)*elementPos(2) + 2*nx*ny - nx - ny;
            globalDOFs(5) = globalDOFs(4) + 1;
            globalDOFs(6) = globalDOFs(4) + nx;
            globalDOFs(7) = globalDOFs(4) + nx + 1;
            globalDOFs(8) = globalDOFs(0) + 3*nx*ny - nx - ny;
            globalDOFs(9) = globalDOFs(8) + nx - 1;
            globalDOFs(10) = globalDOFs(9) + nx;
            globalDOFs(11) = globalDOFs(9) + 1;
        }
        

        return globalDOFs;
    }


    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION std::function<T(size_t,size_t,size_t)> NedelecSpace<T, Dim, Order, ElementType,
                        QuadratureType, FieldType>::curlCurlOperator() const {
        
        static_assert(Dim == 3, "Currently the curl curl operator is only supported for 3D");
        
        
        // The gradient of the basis functions at the different local quadrature
        // points for the reference element. Dimension go over quadrature
        // points, DOFs, dimensions of basis function.
        Kokkos::View<T****> gradBasis("gradBasis", QuadratureType::numElementNodes,
                                        this->numElementDOFs, 3, 3);
        const Vector<point_t, QuadratureType::numElementNodes> qPoints =
            this->quadrature_m.getIntegrationNodesForRefElement();

        Kokkos::parallel_for("ComputeGradBasis", QuadratureType::numElementNodes,
            KOKKOS_CLASS_LAMBDA(const int q){
                if (Dim == 2) {
                    gradBasis(q,0,0,0) = 0;
                    gradBasis(q,0,0,1) = -1;
                    gradBasis(q,0,1,0) = 0;
                    gradBasis(q,0,1,1) = 0;

                    gradBasis(q,1,0,0) = 0;
                    gradBasis(q,1,0,1) = 1;
                    gradBasis(q,1,1,0) = 0;
                    gradBasis(q,1,1,1) = 0;

                    gradBasis(q,2,0,0) = 0;
                    gradBasis(q,2,0,1) = 0;
                    gradBasis(q,2,1,0) = -1;
                    gradBasis(q,2,1,1) = 0;

                    gradBasis(q,3,0,0) = 0;
                    gradBasis(q,3,0,1) = 0;
                    gradBasis(q,3,1,0) = 1;
                    gradBasis(q,3,1,1) = 0;
                } else if (Dim == 3) {
                    T x = qPoints[q][0];
                    T y = qPoints[q][1];
                    T z = qPoints[q][2];

                    gradBasis(q,0,0,0) = 0;
                    gradBasis(q,0,0,1) = -1+z;
                    gradBasis(q,0,0,2) = -1+y;
                    gradBasis(q,0,1,0) = 0;
                    gradBasis(q,0,1,1) = 0;
                    gradBasis(q,0,1,2) = 0;
                    gradBasis(q,0,2,0) = 0;
                    gradBasis(q,0,2,1) = 0;
                    gradBasis(q,0,2,2) = 0;

                    gradBasis(q,1,0,0) = 0;
                    gradBasis(q,1,0,1) = 1-z;
                    gradBasis(q,1,0,2) = -y;
                    gradBasis(q,1,1,0) = 0;
                    gradBasis(q,1,1,1) = 0;
                    gradBasis(q,1,1,2) = 0;
                    gradBasis(q,1,2,0) = 0;
                    gradBasis(q,1,2,1) = 0;
                    gradBasis(q,1,2,2) = 0;

                    gradBasis(q,2,0,0) = 0;
                    gradBasis(q,2,0,1) = -z;
                    gradBasis(q,2,0,2) = 1-y;
                    gradBasis(q,2,1,0) = 0;
                    gradBasis(q,2,1,1) = 0;
                    gradBasis(q,2,1,2) = 0;
                    gradBasis(q,2,2,0) = 0;
                    gradBasis(q,2,2,1) = 0;
                    gradBasis(q,2,2,2) = 0;

                    gradBasis(q,3,0,0) = 0;
                    gradBasis(q,3,0,1) = z;
                    gradBasis(q,3,0,2) = y;
                    gradBasis(q,3,1,0) = 0;
                    gradBasis(q,3,1,1) = 0;
                    gradBasis(q,3,1,2) = 0;
                    gradBasis(q,3,2,0) = 0;
                    gradBasis(q,3,2,1) = 0;
                    gradBasis(q,3,2,2) = 0;

                    gradBasis(q,4,0,0) = 0;
                    gradBasis(q,4,0,1) = 0;
                    gradBasis(q,4,0,2) = 0;
                    gradBasis(q,4,1,0) = -1+z;
                    gradBasis(q,4,1,1) = 0;
                    gradBasis(q,4,1,2) = -1+x;
                    gradBasis(q,4,2,0) = 0;
                    gradBasis(q,4,2,1) = 0;
                    gradBasis(q,4,2,2) = 0;

                    gradBasis(q,5,0,0) = 0;
                    gradBasis(q,5,0,1) = 0;
                    gradBasis(q,5,0,2) = 0;
                    gradBasis(q,5,1,0) = 1-z;
                    gradBasis(q,5,1,1) = 0;
                    gradBasis(q,5,1,2) = -x;
                    gradBasis(q,5,2,0) = 0;
                    gradBasis(q,5,2,1) = 0;
                    gradBasis(q,5,2,2) = 0;

                    gradBasis(q,6,0,0) = 0;
                    gradBasis(q,6,0,1) = 0;
                    gradBasis(q,6,0,2) = 0;
                    gradBasis(q,6,1,0) = -z;
                    gradBasis(q,6,1,1) = 0;
                    gradBasis(q,6,1,2) = 1-x;
                    gradBasis(q,6,2,0) = 0;
                    gradBasis(q,6,2,1) = 0;
                    gradBasis(q,6,2,2) = 0;

                    gradBasis(q,7,0,0) = 0;
                    gradBasis(q,7,0,1) = 0;
                    gradBasis(q,7,0,2) = 0;
                    gradBasis(q,7,1,0) = z;
                    gradBasis(q,7,1,1) = 0;
                    gradBasis(q,7,1,2) = x;
                    gradBasis(q,7,2,0) = 0;
                    gradBasis(q,7,2,1) = 0;
                    gradBasis(q,7,2,2) = 0;

                    gradBasis(q,8,0,0) = 0;
                    gradBasis(q,8,0,1) = 0;
                    gradBasis(q,8,0,2) = 0;
                    gradBasis(q,8,1,0) = 0;
                    gradBasis(q,8,1,1) = 0;
                    gradBasis(q,8,1,2) = 0;
                    gradBasis(q,8,2,0) = -1+y;
                    gradBasis(q,8,2,1) = -1+x;
                    gradBasis(q,8,2,2) = 0;

                    gradBasis(q,9,0,0) = 0;
                    gradBasis(q,9,0,1) = 0;
                    gradBasis(q,9,0,2) = 0;
                    gradBasis(q,9,1,0) = 0;
                    gradBasis(q,9,1,1) = 0;
                    gradBasis(q,9,1,2) = 0;
                    gradBasis(q,9,2,0) = 1-y;
                    gradBasis(q,9,2,1) = -x;
                    gradBasis(q,9,2,2) = 0;

                    gradBasis(q,10,0,0) = 0;
                    gradBasis(q,10,0,1) = 0;
                    gradBasis(q,10,0,2) = 0;
                    gradBasis(q,10,1,0) = 0;
                    gradBasis(q,10,1,1) = 0;
                    gradBasis(q,10,1,2) = 0;
                    gradBasis(q,10,2,0) = -y;
                    gradBasis(q,10,2,1) = 1-x;
                    gradBasis(q,10,2,2) = 0;

                    gradBasis(q,11,0,0) = 0;
                    gradBasis(q,11,0,1) = 0;
                    gradBasis(q,11,0,2) = 0;
                    gradBasis(q,11,1,0) = 0;
                    gradBasis(q,11,1,1) = 0;
                    gradBasis(q,11,1,2) = 0;
                    gradBasis(q,11,2,0) = y;
                    gradBasis(q,11,2,1) = x;
                    gradBasis(q,11,2,2) = 0;
                }
            }
        );
        


        auto f = [gradBasis] (size_t i, size_t j, size_t q) -> T {
            point_t a = {gradBasis(q,j,2,1) - gradBasis(q,j,1,2),
                      gradBasis(q,j,0,2) - gradBasis(q,j,2,0),
                      gradBasis(q,j,1,0) - gradBasis(q,j,0,1)};
            
            point_t b = {gradBasis(q,i,2,1) - gradBasis(q,i,1,2),
                      gradBasis(q,i,0,2) - gradBasis(q,i,2,0),
                      gradBasis(q,i,1,0) - gradBasis(q,i,0,1)};


            return a.dot(b);
        };

        

        return f;
        
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>::point_t
        NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
            ::getLocalDOFPosition(size_t localDOFIndex) const {
            
        point_t position(0);

        switch (localDOFIndex) {
            case 0: position(0) = 0.5; break;
            case 1: position(1) = 0.5; break;
            case 2: position(0) = 0.5; position(1) = 1;   break;
            case 3: position(0) = 1;   position(1) = 0.5; break;
        }

        return position;
    }


    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION std::function<T(size_t,size_t,size_t)> NedelecSpace<T, Dim, Order, ElementType,
                        QuadratureType, FieldType>::massOperator() const {
        
        // The values of the basis functions at the different qudarature points.
        // Dimensions go over quadrature points, DOFs.
        Vector<Vector<point_t, this->numElementDOFs>,
                    QuadratureType::numElementNodes> basisFunctions;

        const Vector<point_t, QuadratureType::numElementNodes> qPoints =
            this->quadrature_m.getIntegrationNodesForRefElement();

        
        for (size_t q = 0; q < QuadratureType::numElementNodes; ++q) {
            if (Dim == 2) {
                T x = qPoints[q][0];
                T y = qPoints[q][1];
                
                basisFunctions[q][0] = {1.-y,0.};
                basisFunctions[q][1] = {y, 0.};
                basisFunctions[q][2] = {0.,1.-x};
                basisFunctions[q][3] = {0.,x};
            } else {
                T x = qPoints[q][0];
                T y = qPoints[q][1];
                T z = qPoints[q][2];
                
                basisFunctions[q][0] = {y*z-y-z+1.,0.,0.};
                basisFunctions[q][1] = {y*(1.-z),0.,0.};
                basisFunctions[q][2] = {z*(1.-y),0.,0.};
                basisFunctions[q][3] = {y*z,0.,0.};
                basisFunctions[q][4] = {0.,x*z-x-z+1.,0.};
                basisFunctions[q][5] = {0.,x*(1.-z),0.};
                basisFunctions[q][6] = {0.,z*(1.-x),0.};
                basisFunctions[q][7] = {0.,x*z,0.};
                basisFunctions[q][8] = {0.,0.,x*y-x-y+1.};
                basisFunctions[q][9] = {0.,0.,x*(1.-y)};
                basisFunctions[q][10] = {0.,0.,y*(1.-x)};
                basisFunctions[q][11] = {0.,0.,x*y};
            }
        }
        


        auto f = [basisFunctions] (size_t i, size_t j, size_t q) -> T {

            return basisFunctions[q][j].dot(basisFunctions[q][i]);
        };
        
        return f;

    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    template <typename Functor>
    KOKKOS_FUNCTION std::function<T(size_t,size_t,ippl::Vector<T,Dim>)> NedelecSpace<T, Dim, Order,
                        ElementType, QuadratureType, FieldType>::loadOperator(Functor f) const {
        
        ippl::Vector<ippl::Vector<point_t, this->numElementDOFs>,
                        QuadratureType::numElementNodes> basisFunctions;

        const Vector<point_t, QuadratureType::numElementNodes> qPoints =
            this->quadrature_m.getIntegrationNodesForRefElement();

        
        for (size_t q = 0; q < QuadratureType::numElementNodes; ++q) {
            if (Dim == 2) {
                T x = qPoints[q][0];
                T y = qPoints[q][1];
                
                basisFunctions[q][0] = {1.-y,0.};
                basisFunctions[q][1] = {y, 0.};
                basisFunctions[q][2] = {0.,1.-x};
                basisFunctions[q][3] = {0.,x};
            } else {
                T x = qPoints[q][0];
                T y = qPoints[q][1];
                T z = qPoints[q][2];
                
                basisFunctions[q][0] = {y*z-y-z+1.,0.,0.};
                basisFunctions[q][1] = {y*(1.-z),0.,0.};
                basisFunctions[q][2] = {z*(1.-y),0.,0.};
                basisFunctions[q][3] = {y*z,0.,0.};
                basisFunctions[q][4] = {0.,x*z-x-z+1.,0.};
                basisFunctions[q][5] = {0.,x*(1.-z),0.};
                basisFunctions[q][6] = {0.,z*(1.-x),0.};
                basisFunctions[q][7] = {0.,x*z,0.};
                basisFunctions[q][8] = {0.,0.,x*y-x-y+1.};
                basisFunctions[q][9] = {0.,0.,x*(1.-y)};
                basisFunctions[q][10] = {0.,0.,y*(1.-x)};
                basisFunctions[q][11] = {0.,0.,x*y};
            }
        }
        


        auto loadF = [basisFunctions,f] (size_t i, size_t q, point_t x) -> T {
            return basisFunctions[q][i].dot(f(x));
        };


        return loadF;

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

        // start a timer
        static IpplTimings::TimerRef evalAx = IpplTimings::getTimer("evaluateAx");
        IpplTimings::startTimer(evalAx);


        // create a new field for result with view initialized to zero (views are initialized to
        // zero by default)
        FEMVector<T> resultVector = x.deepCopy();
        resultVector = 0;

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w =
            this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q =
            this->quadrature_m.getIntegrationNodesForRefElement();

        // TODO move outside of evaluateAx (I think it is possible for other problems as well)
        // Gradients of the basis functions for the DOF at the quadrature nodes
        
        Vector<Vector<point_t, this->numElementDOFs>, QuadratureType::numElementNodes> curl_b_q;
        Vector<Vector<point_t, this->numElementDOFs>, QuadratureType::numElementNodes> val_b_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
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

        // start a timer
        static IpplTimings::TimerRef outer_loop = IpplTimings::getTimer("evaluateAx: outer loop");
        IpplTimings::startTimer(outer_loop);


        // information to translate the ndindices to FEMVector positions.
        // We add the plus one, because we have that last does not correspond
        // to the position end+1, but simply to end
        ippl::Vector<size_t, Dim> extents = (ldom.last()+1) - ldom.first() + 2;
        ippl::Vector<size_t, Dim> v(1);
        v(1) = 2*extents[0] - 1;

        // Loop over elements to compute contributions
        Kokkos::parallel_for(
            "Loop over elements", policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t index) {
                const size_t elementIndex                            = elementIndices(index);
                const Vector<size_t, this->numElementDOFs> local_dof = this->getLocalDOFIndices();
                const Vector<size_t, this->numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);

                // local DOF indices
                size_t i, j;

                // Element matrix
                Vector<Vector<T, this->numElementDOFs>, this->numElementDOFs> A_K;

                // 1. Compute the Galerkin element matrix A_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    for (j = 0; j < this->numElementDOFs; ++j) {
                        A_K[i][j] = 0.0;
                        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                            size_t I = global_dofs[i];
                            size_t J = global_dofs[j];
                            bool onBoundary = this->isDOFOnBoundary(I) || this->isDOFOnBoundary(J); //this->isDOFOnBoundary(I) && this->isDOFOnBoundary(J) && I == J;
                            A_K[i][j] += w[k] * evalFunction(i, j, curl_b_q[k], val_b_q[k], onBoundary);
                        }
                    }
                }

                // global DOF n-dimensional indices (Vector of N indices
                // representing indices in each dimension)
                size_t I, J;
                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    I = global_dofs[i];

                    // Skip boundary DOFs (Zero Dirichlet BCs)
                    /*
                    if (this->isDOFOnBoundary(I)) {
                        continue;
                    }
                        */
                        
                    
                    

                    // get the appropriate index for the Kokkos view of the
                    // field

                    for (j = 0; j < this->numElementDOFs; ++j) {
                        J = global_dofs[j];

                        // Skip boundary DOFs (Zero Dirichlet BCs)
                        /*
                        if (this->isDOFOnBoundary(J)) {
                            continue;
                        }
                            */
                            
                        
                    

                        // get the appropriate index for the Kokkos view of the
                        // field
                        resultView(I) += A_K[i][j] * view(J);
                    }
                }
            });
        IpplTimings::stopTimer(outer_loop);

        IpplTimings::stopTimer(evalAx);
        
        return resultVector;
    
    }


    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
                typename QuadratureType, typename FieldType>
    FEMVector<T> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::evaluateLoadVector(const FEMVector<NedelecSpace<T, Dim, Order, ElementType,
                                QuadratureType, FieldType>::point_t>& f) const {
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
        Vector<Vector<point_t, this->numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        Vector<Vector<T, this->numElementDOFs>, QuadratureType::numElementNodes>
            quadratureDOFDistances;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                point_t dofPos = getLocalDOFPosition(i);
                point_t d = dofPos - q[k];
                quadratureDOFDistances[k][i] = Kokkos::sqrt(d.dot(d));
            }
        }

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Get domain information and ghost cells
        auto ldom        = layout_m.getLocalNDIndex();


        // Get boundary conditions from field
        FEMVector<T> resultVector = f.template skeletonCopy<T>();
        resultVector = 0;

        // Get field data and make it atomic,
        // since it will be added to during the kokkos loop
        // We work with a temporary field since we need to use field
        // to evaluate the load vector; then we assign temp to RHS field
        AtomicViewType atomic_view = resultVector.getView();
        typename detail::ViewType<point_t, 1>::view_type view = f.getView(); 

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
                const Vector<size_t, this->numElementDOFs> local_dofs  = this->getLocalDOFIndices();
                const Vector<size_t, this->numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);

                size_t i, I;

                // 1. Compute b_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    I = global_dofs[i];

                    
                    if (this->isDOFOnBoundary(I)) {
                        continue;
                    }
                    
                        

                    // calculate the contribution of this element
                    T contrib = 0;
                    for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                        // We now have to interpolate the value of the field
                        // given at the DOF positions to the quadrature point
                        point_t interpolatedVal(0);
                        T distSum = 0;
                        for (size_t j = 0; j < this->numElementDOFs; ++j) {
                            // get field index corresponding to this DOF
                            size_t J = global_dofs[j];
                            T dist = quadratureDOFDistances[k][j];
                            distSum += 1./dist;

                            // get field value at DOF and interpolate to q_k
                            interpolatedVal += 1./dist * view(J);
                        }
                        interpolatedVal /= distSum;

                        contrib += w[k] * basis_q[k][i].dot(interpolatedVal) * absDetDPhi;
                    }

                    // add the contribution of the element to the field
                    atomic_view(I) += contrib;

                }
            });

        IpplTimings::stopTimer(outer_loop);
        IpplTimings::stopTimer(evalLoadV);
        
        return resultVector;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
                typename QuadratureType, typename FieldType>
    template <typename F>
    FEMVector<T> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::evaluateLoadVectorFunctor(const FEMVector<NedelecSpace<T, Dim, Order, ElementType,
                                QuadratureType, FieldType>::point_t>& model, const F& f) const {
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
        Vector<Vector<point_t, this->numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        Vector<point_t, QuadratureType::numElementNodes> rhsValues;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            rhsValues[k] = f(q[k]);
        }

        // Absolute value of det Phi_K
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(
            this->getElementMeshVertexPoints(zeroNdIndex)));

        // Get domain information and ghost cells
        auto ldom        = layout_m.getLocalNDIndex();


        // Get boundary conditions from field
        FEMVector<T> resultVector = model.template skeletonCopy<T>();
        resultVector = 0;

        // Get field data and make it atomic,
        // since it will be added to during the kokkos loop
        // We work with a temporary field since we need to use field
        // to evaluate the load vector; then we assign temp to RHS field
        AtomicViewType atomic_view = resultVector.getView();

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
                const Vector<size_t, this->numElementDOFs> local_dofs  = this->getLocalDOFIndices();
                const Vector<size_t, this->numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);

                size_t i, I;

                // 1. Compute b_K
                for (i = 0; i < this->numElementDOFs; ++i) {
                    I = global_dofs[i];

                    
                    if (this->isDOFOnBoundary(I)) {
                        //int side  = getBoundarySide(I);
                        //atomic_view(I) = -10*(side+1);
                        continue;
                    }
                        
                        
                        

                    // calculate the contribution of this element
                    T contrib = 0;
                    for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                        // We now have to interpolate the value of the field
                        // given at the DOF positions to the quadrature point
                        
                        point_t pos = this->ref_element_m.localToGlobal(
                            this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex)),
                            q[k]); 
                        
                        point_t interpolatedVal = f(pos);

                        contrib += w[k] * basis_q[k][i].dot(interpolatedVal) * absDetDPhi;
                    }

                    // add the contribution of the element to the field
                    atomic_view(I) += contrib;
                
                }    
            });
            

        IpplTimings::stopTimer(outer_loop);
        IpplTimings::stopTimer(evalLoadV);
        
        return resultVector;
    }



    ///////////////////////////////////////////////////////////////////////
    /// Basis functions and gradients /////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
                typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>::point_t
                            NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                            ::evaluateRefElementShapeFunction(const size_t& localDOF,
                                const NedelecSpace<T, Dim, Order, ElementType,
                                    QuadratureType, FieldType>::point_t& localPoint) const {

        static_assert(Order == 1, "Only order 1 is supported at the moment");
        // Assert that the local vertex index is valid.
        assert(localDOF < this->numElementDOFs && "The local vertex index is invalid"); 

        assert(this->ref_element_m.isPointInRefElement(localPoint)
            && "Point is not in reference element");

        point_t result(0);

        if (Dim == 2) {
            T x = localPoint(0);
            T y = localPoint(1);

            switch (localDOF){
                case 0: result(0) = 1 - y; break;
                case 1: result(1) = 1 - x; break;
                case 2: result(0) = y; break;
                case 3: result(1) = x; break;
            }
        } else if (Dim == 3) {
            T x = localPoint(0);
            T y = localPoint(1);
            T z = localPoint(2);

            switch (localDOF){
                case 0: result(0) = y*z - y - z + 1; break;
                case 1: result(1) = x*z - x - z + 1; break;
                case 2: result(0) = y*(1 - z); break;
                case 3: result(1) = x*(1 - z); break;
                case 4: result(2) = x*y - x - y + 1; break;
                case 5: result(2) = x*(1 - y); break;
                case 6: result(2) = x*y; break;
                case 7: result(2) = y*(1 - x); break;
                case 8: result(0) = z*(1 - y); break;
                case 9: result(1) = z*(1 - x); break;
                case 10: result(0) = y*z; break;
                case 11: result(1) = x*z; break;
            }
        }


        return result;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION typename NedelecSpace<T, Dim, Order, ElementType,
                                QuadratureType, FieldType>::point_t
                             NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::evaluateRefElementShapeFunctionGradient(const size_t& localDOF,
                                    const NedelecSpace<T, Dim, Order, ElementType,
                                        QuadratureType, FieldType>::point_t& localPoint) const {

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

        // To construct the gradient we need to loop over the dimensions and
        // multiply the shape functions in each dimension except the current
        // one. The one of the current dimension is replaced by the derivative
        // of the shape function in that dimension, which is either 1 or -1.
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

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    KOKKOS_FUNCTION typename NedelecSpace<T, Dim, Order, ElementType,
                                QuadratureType, FieldType>::point_t
                             NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::evaluateRefElementShapeFunctionCurl(const size_t& localDOF,
                                    const NedelecSpace<T, Dim, Order, ElementType,
                                        QuadratureType, FieldType>::point_t& localPoint) const {
        
        point_t result(0);

        if (Dim == 2) {
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
            /*
            if (localDOF % 2 == 0) {
                // are on edge which is parallel to x-axis, therefore x
                // derivative is zero and y derivative is based on if we are at
                // the lower or upper edge.
                result(0) = -(1 - 2*(localDOF == 0));
            } else {
                // are on edge which is parallel to y-axis, therefore y
                // derivative is zero and x derivative is bases on if we are at
                // the left or right edge
                result(0) = (1 - 2*(localDOF == 1));
            }
            */
        } else {

        }

        return result;
        
    }


    ///////////////////////////////////////////////////////////////////////
    /// FEMVector conversion //////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    
    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    FEMVector<T> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::interpolateToFEMVector(const FieldType& field) const {
        

        
        // Next up we need to create the neighbor thing and get all the indices
        auto& layout = field.getLayout();
        auto neighbors = layout.getNeighbors();
        auto neighborSendRange = layout.getNeighborsSendRange();
        auto neighborRecvRange = layout.getNeighborsRecvRange();
        std::vector<size_t> neighborsFV;
        std::vector< Kokkos::View<size_t*> > sendIdxs;
        std::vector< Kokkos::View<size_t*> > recvIdxs;
        std::vector< std::vector<size_t> > sendIdxsTemp;
        std::vector< std::vector<size_t> > recvIdxsTemp;

        /*
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

                if (Dim == 1) {
                    for (size_t x = recvRange.lo[0]; x < recvRange.hi[0]; x++) {
                        recvIdxsTemp[idx].push_back(x);
                    }
                } else if (Dim == 2) {
                    for (size_t x = recvRange.lo[0]; x < recvRange.hi[0]; x++) {
                        for (size_t y = recvRange.lo[1]; y < recvRange.hi[1]; y++) {
                            recvIdxsTemp[idx].push_back(x*v[0] + y*v[1]);
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
        */

        // calculate the number of elements which the FEMVector is going to have
        size_t n = 1;
        if constexpr (Dim == 2) {
            size_t nx = this->nr_m[0];
            size_t ny = this->nr_m[1];
            n = nx*(ny-1) + ny*(nx-1);
        } else if constexpr (Dim == 3) {
            
        }

        // We only handle the 2D case here
        if (Dim == 2) {
            auto& view = field.getView();
            // Now we go though all the points and for each of them get the
            // appropriate DOFs.

        }

    }


    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    FEMVector<Vector<T, Dim> > NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::reconstructBasis(const FEMVector<T>& coef) const {
        
        // Loop over all the global degrees of freedom
        FEMVector<Vector<T,Dim>> outVector = coef.template skeletonCopy<Vector<T,Dim>>();
        size_t nx = this->nr_m[0];

        auto coefView = coef.getView();
        auto outView = outVector.getView();
        
        size_t n = coef.size();
        Kokkos::parallel_for("reconstructBasis", n,
            KOKKOS_CLASS_LAMBDA(size_t i){
                // In order to do this we need to figure out to which axis we
                // are parallel
                bool onXAxis = i - (2*nx-1) * (i / (2*nx - 1)) < (nx-1);
                if (onXAxis) {
                    outView(i)[0] = coefView(i);
                    T yVal = 0;
                    int bval = getBoundarySide(i);
                    switch (bval) {
                    case -1:
                        yVal += coefView(i - nx);
                        yVal += coefView(i - (nx-1));
                        yVal += coefView(i + nx-1);
                        yVal += coefView(i + nx);
                        yVal *= 0.25;
                        break;
                    case 0:
                        yVal += coefView(i + nx-1);
                        yVal += coefView(i + nx);
                        yVal *= 0.5;
                        break;
                    case 2:
                        yVal += coefView(i - nx);
                        yVal += coefView(i - (nx-1));
                        yVal *= 0.5;
                        break;
                    default:
                        IpplException("reconstructBasis", "Wrong boundary");
                        break;
                    }
                    outView(i)[1] = yVal;

                } else {
                    outView(i)[1] = coefView(i);
                    T xVal = 0;
                    int bval = getBoundarySide(i);
                    switch (bval) {
                    case -1:
                        xVal += coefView(i - nx);
                        xVal += coefView(i + nx-1);
                        xVal += coefView(i - (nx-1));
                        xVal += coefView(i + nx);
                        xVal *= 0.25;
                        break;
                    case 1:
                        xVal += coefView(i - (nx-1));
                        xVal += coefView(i + nx);
                        xVal *= 0.5;
                        break;
                    case 3:
                        xVal += coefView(i - nx);
                        xVal += coefView(i + nx-1);
                        xVal *= 0.5;
                        break;
                    default:
                        IpplException("reconstructBasis", "Wrong boundary");
                        break;
                    }
                    outView(i)[0] = xVal;
                }
                
                
            }
        );
        
        return outVector;
    }



    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    void NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::reconstructSolution(const FEMVector<T>& x,
                                    FieldType& field) const {
        
        
        // Loop over all the global degrees of freedom
        size_t nx = this->nr_m[0];
        size_t ny = this->nr_m[1];

        auto coefView = x.getView();
        auto outView = field.getView();
        
        size_t n = x.size();
        Kokkos::parallel_for("reconstructBasis", n,
            KOKKOS_CLASS_LAMBDA(size_t i){
                // In order to do this we need to figure out to which axis we
                // are parallel
                size_t y = i / (2*nx - 1);
                bool onXAxis = i - (2*nx-1) * y < (nx-1);
                if (onXAxis) {
                    size_t fieldIdx1 = i - y*(nx-1);
                    auto fieldNDIdx1 = this->getMeshVertexNDIndex(fieldIdx1);
                    size_t fieldIdx2 = i + 1 - y*(nx-1);
                    auto fieldNDIdx2 = this->getMeshVertexNDIndex(fieldIdx2);
                    
                    T factor = fieldNDIdx1[0] == 0 || fieldNDIdx1[0] == nx-1 ? 1.0 : 1.0;//0.5;
                    apply(outView,fieldNDIdx1)[0] = factor*coefView(i);

                    factor = fieldNDIdx2[0] == 0 || fieldNDIdx2[0] == nx-1 ? 1.0 : 1.0;//0.5;
                    apply(outView,fieldNDIdx2)[0] = factor*coefView(i);

                } else {
                    size_t fieldIdx1 = i - (y+1)*(nx-1);
                    size_t fieldIdx2 = i + 1 - y*(nx-1);
                    
                    auto fieldNDIdx1 = this->getMeshVertexNDIndex(fieldIdx1);
                    auto fieldNDIdx2 = this->getMeshVertexNDIndex(fieldIdx2);
                    
                    T factor = fieldNDIdx1[1] == 0 || fieldNDIdx1[1] == ny-1 ? 1.0 : 1.0;//0.5;
                    apply(outView,fieldNDIdx1)[1] = factor*coefView(i);

                    factor = fieldNDIdx2[1] == 0 || fieldNDIdx2[1] == ny-1 ? 1.0 : 1.0;//0.5;
                    apply(outView,fieldNDIdx2)[1] = factor*coefView(i);
                }
                
                
            }
        );
                                    
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,
              typename QuadratureType, typename FieldType>
    template <typename F>
    T NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::computeError(const FEMVector<Vector<T,Dim> >& u_h, const F& u_sol) const {
        
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
        Vector<Vector<point_t, this->numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        Vector<Vector<T, this->numElementDOFs>, QuadratureType::numElementNodes> 
            quadratureDOFDistances;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                point_t dofPos = getLocalDOFPosition(i);
                point_t d = dofPos - q[k];
                quadratureDOFDistances[k][i] = Kokkos::sqrt(d.dot(d));
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
                const Vector<size_t, this->numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);

                // contribution of this element to the error
                T contrib = 0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    point_t val_u_sol = u_sol(this->ref_element_m.localToGlobal(
                        this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex)),
                            q[k]));

                    point_t val_u_h = 0;
                    T distSum = 0;
                    for (size_t j = 0; j < this->numElementDOFs; ++j) {
                        // get field index corresponding to this DOF
                        size_t J = global_dofs[j];
                        T dist = quadratureDOFDistances[k][j];
                        distSum += 1./dist;

                        // get field value at DOF and interpolate to q_k
                        val_u_h += 1./dist * view(J);
                    }
                    val_u_h /= distSum;

                    point_t dif = (val_u_sol -  val_u_h) ;
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
    template <typename F>
    T NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::computeErrorCoeff(const FEMVector<T>& u_h, const F& u_sol) const {
        
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
        Vector<Vector<point_t, this->numElementDOFs>, QuadratureType::numElementNodes> basis_q;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                basis_q[k][i] = this->evaluateRefElementShapeFunction(i, q[k]);
            }
        }

        Vector<Vector<point_t, this->numElementDOFs>, QuadratureType::numElementNodes> 
            quadratureDOFDistances;
        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
            for (size_t i = 0; i < this->numElementDOFs; ++i) {
                point_t dofPos = getLocalDOFPosition(i);
                point_t d = dofPos - q[k];
                quadratureDOFDistances[k][i] = d;
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

    
        size_t nx = this->nr_m[0];
        // Loop over elements to compute contributions
        Kokkos::parallel_reduce("Compute error over elements",
            policy_type(0, elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(size_t index, double& local) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, this->numElementDOFs> global_dofs =
                    this->getGlobalDOFIndices(elementIndex);

                // contribution of this element to the error
                T contrib = 0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    point_t val_u_sol = u_sol(this->ref_element_m.localToGlobal(
                        this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex)),
                            q[k]));

                    point_t val_u_h = 0;
                    point_t distSum = 0;
                    for (size_t j = 0; j < this->numElementDOFs; ++j) {
                        // get field index corresponding to this DOF
                        bool onXAxis = j == 0 || j == 2;
                        size_t J = global_dofs[j];
                        point_t dist = quadratureDOFDistances[k][j];

                        // get field value at DOF and interpolate to q_k
                        if (onXAxis) {
                            val_u_h(0) += (1-Kokkos::abs(dist[1])) * view(J);
                        }else {
                            val_u_h(1) += (1-Kokkos::abs(dist[0])) * view(J);
                        }
                    }
                    //val_u_h /= distSum;

                    point_t dif = (val_u_sol -  val_u_h) ;
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
    template <typename F>
    T NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldType>
                                ::computeErrorInf(const FieldType& u_h, const F& u_sol) const {
        if (this->quadrature_m.getOrder() < (2 * Order + 1)) {
            // throw exception
            throw IpplException("NedelecSpace::computeError()",
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
        Kokkos::parallel_reduce("Compute error over elements",
            policy_type(0, elementIndices.extent(0)),
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
            Kokkos::Max<double>(error)
        );

        // MPI reduce
        T global_error = 0.0;
        Comm->allreduce(error, global_error, 1, std::greater<T>());

        return global_error;
    }

}  // namespace ippl
