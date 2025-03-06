
namespace ippl {

    // NedelecSpace constructor, which calls the FiniteElementSpace constructor,
    // and decomposes the elements among ranks according to layout.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::NedelecSpace(UniformCartesian<T, Dim>& mesh, ElementType& ref_element, const QuadratureType& quadrature, const Layout_t& layout) : FiniteElementSpace<T, Dim, getNedelecNumElementDOFs(Dim, Order), ElementType, QuadratureType, FieldLHS, FieldRHS>(mesh, ref_element, quadrature) {
        // Assert that the dimension is either 2 or 3.
        static_assert(Dim >= 2 && Dim <= 3,"The Nedelec Finite Element space only supports 2D and 3D meshes");

        // Initialize the elementIndices view
        initializeElementIndices(layout);
    }

    // NedelecSpace constructor, which calls the FiniteElementSpace constructor.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::NedelecSpace(UniformCartesian<T, Dim>& mesh, ElementType& ref_element, const QuadratureType& quadrature) : FiniteElementSpace<T, Dim, getNedelecNumElementDOFs(Dim, Order), ElementType, QuadratureType, FieldLHS, FieldRHS>(mesh, ref_element, quadrature) {
        // Assert that the dimension is either 2 or 3.
        static_assert(Dim >= 2 && Dim <= 3,"The Nedelec Finite Element space only supports 2D and 3D meshes");
    }

    // NedelecSpace initializer, to be made available to the FEMPoissonSolver 
    // such that we can call it from setRhs.
    // Sets the correct mesh ad decomposes the elements among ranks according to layout.
    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::initialize(UniformCartesian<T, Dim>& mesh, const Layout_t& layout) {
        FiniteElementSpace<T, Dim, getNedelecNumElementDOFs(Dim, Order), ElementType, QuadratureType, FieldLHS, FieldRHS>::setMesh(mesh);

        // Initialize the elementIndices view
        initializeElementIndices(layout);
    }

    // Initialize element indices Kokkos View
    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    void NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::initializeElementIndices(const Layout_t& layout) {
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
        Kokkos::parallel_reduce("ComputePoints", npoints, KOKKOS_CLASS_LAMBDA(const int i, int& local) {
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

        Kokkos::parallel_for("RemoveNaNs", npoints, KOKKOS_CLASS_LAMBDA(const int i) {
                if ((points(i) != 0) || (i == 0)) {
                    const size_t idx    = Kokkos::atomic_fetch_add(&index(), 1);
                    elementIndices(idx) = points(i);
                }
            });
    }

    ///////////////////////////////////////////////////////////////////////
    /// Degree of Freedom operations //////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::numGlobalDOFs() const {
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

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    size_t NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::getLocalDOFIndex(const size_t& elementIndex, const size_t& globalDOFIndex) const {
        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3");
        // TODO fix not order independent, only works for order 1
        static_assert(Order == 1, "Only order 1 is supported at the moment");

        // Get all the global DOFs for the element
        const Vector<size_t, this->numElementDOFs> global_dofs = this->getGlobalDOFIndices(elementIndex);

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

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION size_t
    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,FieldRHS>::getGlobalDOFIndex(const size_t& elementIndex, const size_t& localDOFIndex) const {
        const auto global_dofs = this->getGlobalDOFIndices(elementIndex);

        return global_dofs[localDOFIndex];
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    Vector<size_t, NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::numElementDOFs> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::getLocalDOFIndices() const {
        Vector<size_t, numElementDOFs> localDOFs;

        for (size_t dof = 0; dof < numElementDOFs; ++dof) {
            localDOFs[dof] = dof;
        }

        return localDOFs;
    }

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION
    Vector<size_t, NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::numElementDOFs> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::getGlobalDOFIndices(const size_t& elementIndex) const {
        Vector<size_t, this->numElementDOFs> globalDOFs(0);
        indices_t elementPos = this->getElementNDIndex(elementIndex);
        // Here we just hardcode the different global dof positions, it is probably the easiest way:
        if (Dim == 2) {
            size_t nx = this->nr_m[0]-1;
            size_t ny = this->nr_m[1]-1;
            size_t x = elementPos[0];
            size_t y = elementPos[1];

            globalDOFs[0] = y*nx + x;
            globalDOFs[1] = (y+1)*nx + x;
            globalDOFs[2] =     x*ny + y + nx*(ny+1);
            globalDOFs[3] = (x+1)*ny + y + nx*(ny+1);
        }
        else if (Dim == 3) {
            size_t nx = this->nr_m[0]-1;
            size_t ny = this->nr_m[1]-1;
            size_t nz = this->nr_m[2]-1;
            size_t x = elementPos[0];
            size_t y = elementPos[1];
            size_t z = elementPos[2];

            globalDOFs[0]  =     z*((nx+1)*ny + nx*(ny+1)) +     y*nx + x;
            globalDOFs[1]  =     z*((nx+1)*ny + nx*(ny+1)) + (y+1)*nx + x;
            globalDOFs[2]  = (z+1)*((nx+1)*ny + nx*(ny+1)) +     y*nx + x;
            globalDOFs[3]  = (z+1)*((nx+1)*ny + nx*(ny+1)) + (y+1)*nx + x;
            globalDOFs[4]  =     z*((nx+1)*ny + nx*(ny+1)) +     x*ny + y + nx*(ny+1);
            globalDOFs[5]  =     z*((nx+1)*ny + nx*(ny+1)) + (x+1)*ny + y + nx*(ny+1);
            globalDOFs[6]  = (z+1)*((nx+1)*ny + nx*(ny+1)) +     x*ny + y + nx*(ny+1);
            globalDOFs[7]  = (z+1)*((nx+1)*ny + nx*(ny+1)) + (x+1)*ny + y + nx*(ny+1);
            globalDOFs[8]  = (nz+1)*((nx+1)*ny + nx*(ny+1)) +     y*(nx+1)*nz +     x*nz + z;
            globalDOFs[9]  = (nz+1)*((nx+1)*ny + nx*(ny+1)) +     y*(nx+1)*nz + (x+1)*nz + z;
            globalDOFs[10] = (nz+1)*((nx+1)*ny + nx*(ny+1)) + (y+1)*(nx+1)*nz +     x*nz + z;
            globalDOFs[11] = (nz+1)*((nx+1)*ny + nx*(ny+1)) + (y+1)*(nx+1)*nz + (x+1)*nz + z;

        }

        return globalDOFs;
    }


    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION std::function<T(size_t,size_t,size_t)> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::curlCurlOperator() const {
        
        static_assert(Dim == 3, "Currently the curl curl operator is only supported for 3D");
        
        
        // The gradient of the basis functions at the different local quadrature points for the reference element. Dimension go over quadrature points, DOFs,
        // dimensions of basis function.
        Kokkos::View<T****> gradBasis("gradBasis", QuadratureType::numElementNodes, this->numElementDOFs, 3, 3);
        const Vector<point_t, QuadratureType::numElementNodes> qPoints = this->quadrature_m.getIntegrationNodesForRefElement();

        Kokkos::parallel_for("ComputeGradBasis", QuadratureType::numElementNodes, KOKKOS_CLASS_LAMBDA(const int q){
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
        });
        


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

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION std::function<T(size_t,size_t,size_t)> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::massOperator() const {
        
        // The values of the basis functions at the different qudarature points. Dimensions go over quadrature points, DOFs.
        Vector<Vector<point_t, this->numElementDOFs>, QuadratureType::numElementNodes> basisFunctions;

        const Vector<point_t, QuadratureType::numElementNodes> qPoints = this->quadrature_m.getIntegrationNodesForRefElement();

        
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

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename Functor>
    KOKKOS_FUNCTION std::function<T(size_t,size_t,ippl::Vector<T,Dim>)> NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::loadOperator(Functor f) const {
        
        ippl::Vector<ippl::Vector<point_t, this->numElementDOFs>, QuadratureType::numElementNodes> basisFunctions;

        const Vector<point_t, QuadratureType::numElementNodes> qPoints = this->quadrature_m.getIntegrationNodesForRefElement();

        
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
    /// Basis functions and gradients /////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION T
    NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::evaluateRefElementShapeFunction(const size_t& localDOF, const NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,FieldRHS>::point_t& localPoint) const {
        static_assert(Order == 1, "Only order 1 is supported at the moment");
        // Assert that the local vertex index is valid.
        assert(localDOF < this->numElementDOFs && "The local vertex index is invalid"); 

        assert(this->ref_element_m.isPointInRefElement(localPoint) && "Point is not in reference element");

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

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    KOKKOS_FUNCTION typename NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::point_t NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::evaluateRefElementShapeFunctionGradient(const size_t& localDOF, const NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::point_t& localPoint) const {
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

    template <typename T, unsigned Dim, unsigned Order, typename ElementType, typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    T NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS, FieldRHS>::computeError(const FieldLHS& u_h, const F& u_sol) const {
        
        if (this->quadrature_m.getOrder() < (2 * Order + 1)) {
            // throw exception
            throw IpplException( "NedelecSpace::computeError()", "Order of quadrature rule for error computation should be > 2*p + 1");
        }

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w = this->quadrature_m.getWeightsForRefElement();

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
        const T absDetDPhi = Kokkos::abs(this->ref_element_m.getDeterminantOfTransformationJacobian(this->getElementMeshVertexPoints(zeroNdIndex)));

        // Variable to sum the error to
        T error = 0;

        // Get domain information and ghost cells
        auto ldom        = (u_h.getLayout()).getLocalNDIndex();
        const int nghost = u_h.getNghost();

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;

        // Loop over elements to compute contributions
        Kokkos::parallel_reduce("Compute error over elements", policy_type(0, elementIndices.extent(0)), KOKKOS_CLASS_LAMBDA(size_t index, double& local) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, this->numElementDOFs> global_dofs = this->getGlobalDOFIndices(elementIndex);

                // contribution of this element to the error
                T contrib = 0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    T val_u_sol = u_sol(this->ref_element_m.localToGlobal(this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex)),q[k]));

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

    template <typename T, unsigned Dim, unsigned Order, typename ElementType,typename QuadratureType, typename FieldLHS, typename FieldRHS>
    template <typename F>
    T NedelecSpace<T, Dim, Order, ElementType, QuadratureType, FieldLHS,FieldRHS>::computeErrorInf(const FieldLHS& u_h, const F& u_sol) const {
        if (this->quadrature_m.getOrder() < (2 * Order + 1)) {
            // throw exception
            throw IpplException("NedelecSpace::computeError()","Order of quadrature rule for error computation should be > 2*p + 1");
        }

        // List of quadrature weights
        const Vector<T, QuadratureType::numElementNodes> w = this->quadrature_m.getWeightsForRefElement();

        // List of quadrature nodes
        const Vector<point_t, QuadratureType::numElementNodes> q = this->quadrature_m.getIntegrationNodesForRefElement();

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
        Kokkos::parallel_reduce("Compute error over elements", policy_type(0, elementIndices.extent(0)), KOKKOS_CLASS_LAMBDA(size_t index, double& local_max) {
                const size_t elementIndex = elementIndices(index);
                const Vector<size_t, this->numElementDOFs> global_dofs = this->getGlobalDOFIndices(elementIndex);
                auto dof_points = this->getElementMeshVertexPoints(this->getElementNDIndex(elementIndex));

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
