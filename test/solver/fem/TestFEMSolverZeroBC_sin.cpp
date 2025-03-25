// Tests the FEM EM diffusion solver by solving the problem:
//
// curl(curl(E)) + E = [(1+k^2)sin(k*y), (1+k^2)sin(k*z), (1+k^2)sin(k*x)]^T, (x,y,z) in [-1,1]^3
// cross(E,n) = 0 on boundary
//
// Exact solution is E = [sin(k*y), sin(k*z), sin(k*x)]
//
// BCs: Zero curl
//
// Usage:
//    ./TestNedelecZeroCurl --info 5

#include "Ippl.h"

#include "Meshes/Centering.h"
#include "FEM/FEMSolver.h"

#include <fstream>



template <typename T, unsigned Dim>
struct AnalyticSol {
    const T pi = Kokkos::numbers::pi_v<T>;

    KOKKOS_FUNCTION const T operator()(ippl::Vector<T, Dim> x_vec) const {
        T val = 1.0;
        for (unsigned d = 0; d < Dim; d++) {
            val *= Kokkos::sin(pi * x_vec[d]);
        }
        return val;
    }
};


template <typename T, unsigned Dim, typename Space>
struct Bilinear {
    
    Bilinear(const Space& space) : space_m(space) {
        // List of quadrature nodes
        const ippl::Vector<ippl::Vector<T,Dim>, Space::Quadrature_t::numElementNodes> q =
            space_m.quadrature_m.getIntegrationNodesForRefElement();
        
        for (size_t k = 0; k < Space::Quadrature_t::numElementNodes; ++k) {
            for (size_t i = 0; i < space_m.numElementDOFs; ++i) {
                grad_b_q[k][i] = space_m.evaluateRefElementShapeFunctionGradient(i, q[k]);
            }
        }

        const ippl::Vector<size_t, Dim> zeroNdIndex = ippl::Vector<size_t, Dim>(0);

        // We can pass the zeroNdIndex here, since the transformation jacobian does not depend
        // on translation
        const auto firstElementVertexPoints = space_m.getElementMeshVertexPoints(zeroNdIndex);

        // Compute Inverse Transpose Transformation Jacobian ()
        DPhiInvT = space_m.ref_element_m.getInverseTransposeTransformationJacobian(firstElementVertexPoints);
    }

    const Space& space_m;
    ippl::Vector<ippl::Vector<ippl::Vector<T,Dim>, Space::numElementDOFs>, Space::Quadrature_t::numElementNodes> grad_b_q;
    ippl::Vector<T, Space::Quadrature_t::numElementNodes> w;
    ippl::Vector<T,Dim> DPhiInvT;

    KOKKOS_INLINE_FUNCTION const T operator()(size_t i, size_t j, size_t q) const {
        return dot(DPhiInvT*grad_b_q[q][j], DPhiInvT*grad_b_q[q][i]).apply();
    }

};


template <typename T, unsigned Dim, typename Space>
struct Linear {
    Linear(const Space& space) : space_m(space) {
        const ippl::Vector<ippl::Vector<T,Dim>, Space::Quadrature_t::numElementNodes> q =
            space_m.quadrature_m.getIntegrationNodesForRefElement();

        for (size_t k = 0; k < Space::Quadrature_t::numElementNodes; ++k) {
            for (size_t i = 0; i < space_m.numElementDOFs; ++i) {
                basis_q[k][i] = space_m.evaluateRefElementShapeFunction(i, q[k]);
            }
        }
    }

    const Space& space_m;
    ippl::Vector<ippl::Vector<T, Space::numElementDOFs>, Space::Quadrature_t::numElementNodes> basis_q;

    KOKKOS_INLINE_FUNCTION const T operator()(size_t i, size_t q, ippl::Vector<T,Dim> x) const {
        const T pi = Kokkos::numbers::pi_v<T>;

        T val = 1.0;
        for (unsigned d = 0; d < Dim; d++) {
            val *= Kokkos::sin(pi * x[d]);
        }

        val = Dim * pi * pi * val;

        return basis_q[q][i] * val;
    }

};

template <typename T, unsigned Dim>
void testFEMSolver(const unsigned& numNodesPerDim, const T& domain_start = 0.0,
                   const T& domain_end = 1.0) {

    using Tlhs = T;
    using Mesh_t   = ippl::UniformCartesian<T, Dim>;
    using Field_t  = ippl::Field<T, Dim, Mesh_t, Cell>;
    using BConds_t = ippl::BConds<Field_t, Dim>;
    using ElementType = std::conditional_t<Dim == 1, ippl::EdgeElement<T>, std::conditional_t<Dim == 2, ippl::QuadrilateralElement<Tlhs>, ippl::HexahedralElement<Tlhs>>>;
    using QuadratureType = ippl::GaussJacobiQuadrature<Tlhs, 5, ElementType>;
    using SpaceType = ippl::LagrangeSpace<T, Dim, 1, ElementType, QuadratureType, Field_t, Field_t>;
    using MeshType = typename Field_t::Mesh_t;


    ElementType refElement_m;
    QuadratureType quadrature_m(refElement_m, 0.0, 0.0);

    // start the timer
    static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("initTest");
    IpplTimings::startTimer(initTimer);

    Inform m("");
    Inform msg2all("", INFORM_ALL_NODES);



    const unsigned numCellsPerDim = numNodesPerDim - 1;

    // Domain: [-1, 1]
    const ippl::Vector<unsigned, Dim> nodesPerDimVec(numNodesPerDim);
    ippl::NDIndex<Dim> domain(nodesPerDimVec);
    ippl::Vector<T, Dim> cellSpacing((domain_end - domain_start) / static_cast<T>(numCellsPerDim));
    ippl::Vector<T, Dim> origin(domain_start);
    Mesh_t mesh(domain, cellSpacing, origin);

    
    // specifies decomposition; here all dimensions are parallel
    std::array<bool, Dim> isParallel;
    isParallel.fill(true);
    ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, domain, isParallel);


    SpaceType space(mesh, refElement_m, quadrature_m, layout);

    Bilinear<T,Dim,SpaceType> bilinear(space);
    Linear<T,Dim, SpaceType> linear(space);

    ippl::FEMSolver<Field_t, Field_t, Bilinear<T,Dim,SpaceType>, Linear<T,Dim,SpaceType>, SpaceType, Mesh_t, QuadratureType> solver(bilinear, linear, space, mesh, layout, quadrature_m);


    // solve the problem
    Field_t result = solver.solve();

    for (int i = 0; i < ippl::Comm->size(); ++i) {
        if (i == ippl::Comm->rank()) {
            std::ofstream file;
            if (i == 0) {
                file.open("sim_out.csv");
                file << "x, y, z, val\n";
            } else {
                file.open("sim_out.csv", std::ios::app);
            }
            // create host view of inidices
            auto h = Kokkos::create_mirror(solver.elementIndices);
            Kokkos::deep_copy(h, solver.elementIndices);

            // create host view of data
            auto v = result.getHostMirror();
            Kokkos::deep_copy(v, result.getView());

            for (size_t n = 0; n < h.extent(0); ++n) {
                // get coordinates
                const size_t elementIndex = h(n);
                const auto elementPos = space.getElementNDIndex(elementIndex);
                const auto vertexPoses = space.getElementMeshVertexNDIndices(elementPos);
                const auto vertecies = space.getElementMeshVertexPoints(elementPos);

                // the coordinates
                for (int d = 0; d < vertexPoses.dim; ++d) {
                    for (int k = 0; k < Dim; ++k) {
                        file << vertecies[d][k] << ", ";
                    }

                    // the value
                    file << ippl::apply(v,vertexPoses[d]) << "\n";
                }
            }

            file.close();
        }
        ippl::Comm->barrier();
    }

    // start the timer
    static IpplTimings::TimerRef errorTimer = IpplTimings::getTimer("computeError");
    IpplTimings::startTimer(errorTimer);
    
    // Compute the error
    AnalyticSol<T, Dim> analytic;
    T relError = space.computeError(result, analytic);

    if (ippl::Comm->rank() == 0) {
        std::cout << std::setw(10) << numNodesPerDim;
        std::cout << std::setw(25) << std::setprecision(16) << cellSpacing[0];
        std::cout << std::setw(25) << std::setprecision(16) << relError;
        std::cout << std::setw(25) << std::setprecision(16) << solver.getResidue();
        std::cout << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
        std::cout << "\n";
    }
    
    IpplTimings::stopTimer(errorTimer);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("");

        using T = double;
        if (ippl::Comm->rank() == 0) {
            std::cout << std::setw(10) << "#n/dim" << std::setw(25) << "cell spacing" << std::setw(25) << "rel error" <<std::setw(25) << "CG residum" << std::setw(25) << "CG interation count\n";
        }
        testFEMSolver<T, 2u>(7, 1.0, 3.0);
        //for (unsigned n = 1 << 3; n <= 1 << 9; n = n << 1) {
        //    testFEMSolver<T, 3>(n, 1.0, 3.0);
        //}


        // print the timers
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
