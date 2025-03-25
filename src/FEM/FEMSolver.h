#ifndef IPPL_FEMSOLVER_H
#define IPPL_FEMSOLVER_H


#include <functional>
#include <fstream>

#include "LinearSolvers/PCG.h"


namespace ippl{

  template<typename FieldLHS, typename FieldRHS, typename BilinearFunctor, typename LinearFunctor, typename Space, typename Mesh, typename QuadratureType>
  class FEMSolver{
    constexpr static unsigned Dim = FieldLHS::dim;
    constexpr static unsigned numGhosts = 1;
    using Tlhs = typename FieldLHS::value_type;
    using ElementType =
            std::conditional_t<Dim == 1, ippl::EdgeElement<Tlhs>,
                               std::conditional_t<Dim == 2, ippl::QuadrilateralElement<Tlhs>,
                                                  ippl::HexahedralElement<Tlhs>>>;
    using T = Tlhs;
    typedef Vector<size_t, Dim> indices_t;
    typedef typename detail::ViewType<T, Dim, Kokkos::MemoryTraits<Kokkos::Atomic>>::view_type AtomicViewType;
    typedef typename detail::ViewType<T, Dim>::view_type ViewType;
    using PCGSolverAlgorithm_t = CG<FieldLHS, FieldLHS, FieldLHS, FieldLHS, FieldLHS, FieldLHS, FieldRHS>;

    typedef ippl::FieldLayout<FieldLHS::dim> Layout_t;
    typedef Vector<T, Dim> point_t;


    public:
    FEMSolver(BilinearFunctor bilinear, LinearFunctor linear, Space space, Mesh mesh, Layout_t layout, QuadratureType quadrature) : bilinear_m(bilinear), linear_m(linear), space_m(space), mesh_m(mesh), layout_m(layout), refElement_m(), quadrature_m(quadrature) {
        params_m.add("output_type", 0b01);
        params_m.add("max_iterations", 1000);
        params_m.add("tolerance", (Tlhs)1e-13);

        initializeElementIndices(layout);

    }


    FieldLHS solve() {

        const Vector<size_t, Dim> zeroNdIndex = Vector<size_t, Dim>(0);

        // We can pass the zeroNdIndex here, since the transformation jacobian does not depend
        // on translation
        const auto firstElementVertexPoints = space_m.getElementMeshVertexPoints(zeroNdIndex);

        // Compute Inverse Transpose Transformation Jacobian ()
        const Vector<Tlhs, Dim> DPhiInvT = refElement_m.getInverseTransposeTransformationJacobian(firstElementVertexPoints);

        // Compute absolute value of the determinant of the transformation jacobian (|det D
        // Phi_K^T* DPhi_K|)
        const Tlhs absDetDPhi = Kokkos::abs(refElement_m.getDeterminantOfTransformationJacobian(firstElementVertexPoints));


        const Vector<point_t, QuadratureType::numElementNodes> q = quadrature_m.getIntegrationNodesForRefElement();
        const Vector<T, QuadratureType::numElementNodes> w = quadrature_m.getWeightsForRefElement();


        const auto algoOperator = [this, w, q, absDetDPhi, firstElementVertexPoints](FieldLHS field) -> FieldLHS {
            
            // start a timer
            static IpplTimings::TimerRef opTimer = IpplTimings::getTimer("operator");
            IpplTimings::startTimer(opTimer);

            for (unsigned r = 0; r < Comm->size(); ++r) {
                if (r == Comm->rank()) {
                    std::ofstream file;
                    if (r == 0) {
                        file.open("lhs.txt");
                        file << "BEFORE FILL\n";
                    } else {
                        file.open("lhs.txt", std::ios::app);
                    }
                    file << "rank: " << r << "\n";
                    auto h = field.getHostMirror();
                    Kokkos::deep_copy(h, field.getView());
                    for (int i = h.extent(0)-1; i >= 0; --i) {
                        for(int j = 0; j < h.extent(1); ++j) {
                            file << std::setw(15) << h(i,j); 
                        }
                        file << "\n";
                    }
                    file.close();
                }
                Comm->barrier();
            }

            field.fillHalo();
            for (unsigned r = 0; r < Comm->size(); ++r) {
                if (r == Comm->rank()) {
                    std::ofstream file;
                    if (r == 0) {
                        file.open("lhs.txt", std::ios::app);
                        file << "AFTER FILL\n";
                    } else {
                        file.open("lhs.txt", std::ios::app);
                    }
                    file << "rank: " << r << "\n";
                    auto h = field.getHostMirror();
                    Kokkos::deep_copy(h, field.getView());
                    for (int i = h.extent(0)-1; i >= 0; --i) {
                        for(int j = 0; j < h.extent(1); ++j) {
                            file << std::setw(15) << h(i,j); 
                        }
                        file << "\n";
                    }
                    file.close();
                }
                Comm->barrier();
            }

            // start evaluateAx
            ViewType view = field.getView();
            FieldLHS resultField(field.get_mesh(), field.getLayout(), numGhosts);
            AtomicViewType resultView = resultField.getView();

            auto ldom = (field.getLayout()).getLocalNDIndex();

            using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
            using policy_type = Kokkos::RangePolicy<exec_space>;

            BConds<FieldLHS, Dim>& bcField = field.getFieldBC();
            FieldBC bcType = bcField[0]->getBCType();

            // Loop over Elements
            Kokkos::parallel_for("Loop over elements", policy_type(0,elementIndices.extent(0)),
            KOKKOS_CLASS_LAMBDA(const size_t n){
                Vector<Vector<T,space_m.numElementDOFs>,space_m.numElementDOFs> A_K;

                size_t i,j;
                // 1. Compute the Galerkin element matrix A_K
                for (i = 0; i < space_m.numElementDOFs; ++i) {
                    for (j = 0; j < space_m.numElementDOFs; ++j) {
                        A_K[i][j] = 0.0;
                        for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                            A_K[i][j] += w[k]*bilinear_m(i, j, k)*absDetDPhi;
                        }
                    }
                }

                // global DOF n-dimensional indices (Vector of N indices representing indices in
                // each dimension)
                indices_t I_nd, J_nd;


                const size_t elementIndex = elementIndices(n);
                const Vector<size_t, space_m.numElementDOFs> local_dof = space_m.getLocalDOFIndices();
                const Vector<size_t, space_m.numElementDOFs> global_dofs = space_m.getGlobalDOFIndices(elementIndex);
                ippl::Vector<indices_t, space_m.numElementDOFs> global_dof_ndindices;
                for (size_t i = 0; i < space_m.numElementDOFs; ++i) {
                    global_dof_ndindices[i] = space_m.getMeshVertexNDIndex(global_dofs[i]);
                }



                // 2. Compute the contribution to resultAx = A*x with A_K
                for (i = 0; i < space_m.numElementDOFs; ++i) {
                    I_nd = global_dof_ndindices[i];

                    // Skip boundary DOFs (Zero Dirichlet BCs)
                    if ((bcType == ZERO_FACE) && (space_m.isDOFOnBoundary(I_nd))) {
                        continue;
                    }

                    // get the appropriate index for the Kokkos view of the field
                    for (unsigned d = 0; d < Dim; ++d) {
                        I_nd[d] = I_nd[d] - ldom[d].first() + numGhosts;
                    }

                    for (j = 0; j < this->space_m.numElementDOFs; ++j) {
                        J_nd = global_dof_ndindices[j];

                        // Skip boundary DOFs (Zero Dirichlet BCs)
                        if ((bcType == ZERO_FACE) && space_m.isDOFOnBoundary(J_nd)) {
                            continue;
                        }

                        // get the appropriate index for the Kokkos view of the field
                        for (unsigned d = 0; d < Dim; ++d) {
                            J_nd[d] = J_nd[d] - ldom[d].first() + numGhosts;
                        }

                        apply(resultView, I_nd) += A_K[i][j] * apply(view, J_nd);
                    }
                }

            });
            // End evaluateAx

            for (unsigned r = 0; r < Comm->size(); ++r) {
                if (r == Comm->rank()) {
                    std::ofstream file;
                    if (r == 0) {
                        file.open("lhs.txt", std::ios::app);
                        file << "BEFORE ACCUMULATEL\n";
                    } else {
                        file.open("lhs.txt", std::ios::app);
                    }
                    file << "rank: " << r << "\n";
                    auto h = resultField.getHostMirror();
                    Kokkos::deep_copy(h, resultField.getView());
                    for (int i = h.extent(0)-1; i >= 0; --i) {
                        for(int j = 0; j < h.extent(1); ++j) {
                            file << std::setw(15) << h(i,j); 
                        }
                        file << "\n";
                    }
                    file.close();
                }
                Comm->barrier();
            }

            resultField.accumulateHalo();

            for (unsigned r = 0; r < Comm->size(); ++r) {
                if (r == Comm->rank()) {
                    std::ofstream file;
                    if (r == 0) {
                        file.open("lhs.txt", std::ios::app);
                        file << "AFTER ACCUMULATEL\n";
                    } else {
                        file.open("lhs.txt", std::ios::app);
                    }
                    file << "rank: " << r << "\n";
                    auto h = resultField.getHostMirror();
                    Kokkos::deep_copy(h, resultField.getView());
                    for (int i = h.extent(0)-1; i >= 0; --i) {
                        for(int j = 0; j < h.extent(1); ++j) {
                            file << std::setw(15) << h(i,j); 
                        }
                        file << "\n";
                    }
                    file.close();
                }
                Comm->barrier();
            }
            
            IpplTimings::stopTimer(opTimer);

            return resultField;
        };


        pcg_algo_m.setOperator(algoOperator);

        FieldLHS test_old_in(mesh_m, layout_m, 1);
        test_old_in = 1.;
        ippl::BConds<FieldLHS, Dim> test_old_bc_field;
        for (unsigned int i = 0; i < 2 * Dim; ++i) {
            test_old_bc_field[i] = std::make_shared<ippl::ZeroFace<FieldLHS>>(i);
        }
        test_old_in.setFieldBC(test_old_bc_field);
        FieldLHS old_out = algoOperator(test_old_in);


        // create the lhs and rhs
        FieldLHS lhs(mesh_m, layout_m, numGhosts);
        FieldRHS rhs(mesh_m, layout_m, numGhosts);

        ippl::BConds<FieldLHS, Dim> bcField;
        for (unsigned int i = 0; i < 2 * Dim; ++i) {
            bcField[i] = std::make_shared<ippl::ZeroFace<FieldLHS>>(i);
        }
        lhs.setFieldBC(bcField);
        rhs.setFieldBC(bcField);

        AtomicViewType atomic_view = rhs.getView();

        BConds<FieldLHS, Dim>& bcField_ = rhs.getFieldBC();
        FieldBC bcType = bcField_[0]->getBCType();
        auto ldom = (rhs.getLayout()).getLocalNDIndex();
        

        using exec_space  = typename Kokkos::View<const size_t*>::execution_space;
        using policy_type = Kokkos::RangePolicy<exec_space>;
        
        // Loop over Elements
        Kokkos::parallel_for("Loop over elements load linear", policy_type(0,elementIndices.extent(0)),
        KOKKOS_CLASS_LAMBDA(const size_t n){
            const size_t elementIndex                              = elementIndices(n);
            const Vector<size_t, this->space_m.numElementDOFs> local_dofs  = this->space_m.getLocalDOFIndices();
            const Vector<size_t, this->space_m.numElementDOFs> global_dofs =
                this->space_m.getGlobalDOFIndices(elementIndex);

            size_t i, I;
            
            indices_t elementPos = space_m.getElementNDIndex(elementIndex);
            const auto vertecies = space_m.getElementMeshVertexPoints(elementPos);

            // 1. Compute b_K
            for (i = 0; i < this->space_m.numElementDOFs; ++i) {
                I = global_dofs[i];

                // TODO fix for higher order
                auto dof_ndindex_I = this->space_m.getMeshVertexNDIndex(I);

                if ((bcType == ZERO_FACE) && (this->space_m.isDOFOnBoundary(dof_ndindex_I))) {
                    continue;
                }

                // calculate the contribution of this element
                T contrib = 0;
                for (size_t k = 0; k < QuadratureType::numElementNodes; ++k) {
                    contrib += w[k]*linear_m(i, k, refElement_m.localToGlobal(vertecies,q[k]))*absDetDPhi;
                }


                // get the appropriate index for the Kokkos view of the field
                for (unsigned d = 0; d < Dim; ++d) {
                    dof_ndindex_I[d] = dof_ndindex_I[d] - ldom[d].first() + numGhosts;
                }

                // add the contribution of the element to the field
                apply(atomic_view, dof_ndindex_I) += contrib;

            }

        });

        rhs.accumulateHalo();
        rhs.fillHalo();

        //if (ippl::Comm->rank() == 0) {
        //    auto h = rhs.getHostMirror();
        //    Kokkos::deep_copy(h, rhs.getView());
        //    for (int i = 0; i < h.extent(0); ++i) {
        //        for(int j = 0; j < h.extent(1); ++j) {
        //            for(int k = 0; k < h.extent(2); ++k) {
        //                std::cout << h(i,j,k) << " ";
        //            }
        //            std::cout << "\n";
        //        }
        //        std::cout << "-------\n";
        //    }
        //}

        // start a timer
        static IpplTimings::TimerRef pcgTimer = IpplTimings::getTimer("pcg");
        IpplTimings::startTimer(pcgTimer);

        //pcg_algo_m(lhs, rhs, params_m);

        lhs.fillHalo();

        IpplTimings::stopTimer(pcgTimer);

        return lhs;
    }



    /**
     * Query how many iterations were required to obtain the solution
     * the last time this solver was used
     * @return Iteration count of last solve
     */
    int getIterationCount() { return pcg_algo_m.getIterationCount(); }

    /**
     * Query the residue
     * @return Residue norm from last solve
     */
    Tlhs getResidue() const { return pcg_algo_m.getResidue(); }

    public:

    KOKKOS_FUNCTION size_t getElementIndex(const indices_t& ndindex) const {
        size_t element_index = 0;

        // This is the number of cells in each dimension. It is one less than the number of
        // vertices in each dimension, which is returned by Mesh::getGridsize().
        ippl::Vector<size_t, Dim> cells_per_dim = space_m.nr_m - 1;

        size_t remaining_number_of_cells = 1;

        for (unsigned int d = 0; d < Dim; ++d) {
            element_index += ndindex[d] * remaining_number_of_cells;
            remaining_number_of_cells *= cells_per_dim[d];
        }

        return element_index;
    }
    

    void initializeElementIndices(const Layout_t& layout) {
        const auto& ldom = layout.getLocalNDIndex();
        int npoints      = ldom.size();
        auto first       = ldom.first();
        auto last        = ldom.last();
        ippl::Vector<double, Dim> bounds;

        for (size_t d = 0; d < Dim; ++d) {
            bounds[d] = space_m.nr_m[d] - 1;
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
        
        for (int r = 0; r < ippl::Comm->size(); ++r) {
            if (r == Comm->rank()) {
                std::cout << "rank: " << r << "\n";
                std::cout << "min: " << first << "\n";
                std::cout << "max: " << last << "\n";
                for (int i = 0; i < elementIndices.extent(0); ++i) {
                    std::cout << i << ": " << elementIndices(i) << "\n";
                } 
            }
            Comm->barrier();
        }
    }



    Space space_m;
    Mesh mesh_m;
    ippl::FieldLayout<Dim> layout_m;
    ElementType refElement_m;
    QuadratureType quadrature_m;
    PCGSolverAlgorithm_t pcg_algo_m;
    BilinearFunctor bilinear_m;
    LinearFunctor linear_m;

    Kokkos::View<size_t*> elementIndices;

    ParameterList params_m;

  };
};


#endif