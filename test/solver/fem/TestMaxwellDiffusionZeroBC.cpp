// Tests the FEM MaxwellDiffuson solver by solving the problem:
//
// curl(curl(E)) + E = f in [1.0,3.0]^dim
// n x E = 0 on boundary 
// with dim = 2,3
// and k = pi
// and f = {(1+k^2)sin(k*y), (1+k^2)sin(k*x)} in 2d
//       = {(1 + k^2)*sin(k*y)*sin(k*z),
//          (1 + k^2)*sin(k*y)*sin(k*z),
//          (1 + k^2)*sin(k*y)*sin(k*z)} in 3d
//
// Exact solution is E = {sin(k*y),sin(k*x)} in 2d
//                     = {sin(k*y)*sin(k*z),
//                        sin(k*y)*sin(k*z),
//                        sin(k*y)*sin(k*z)} in 3d
//
// BCs: Zero dirichlet bc.
//

#include "Ippl.h"

#include "Meshes/Centering.h"
#include "MaxwellSolvers/FEMMaxwellDiffusionSolver.h"

#include <fstream>

template <typename T, unsigned Dim>
struct Analytical{
    using point_t =  ippl::Vector<T, Dim>;

    T k;
    Analytical(T k) : k(k) {}

    KOKKOS_FUNCTION const point_t operator() (const point_t& pos) const{
        point_t sol(0);
        if constexpr (Dim == 2) {
            sol[0] = Kokkos::sin(k*pos[1]);
            sol[1] = Kokkos::sin(k*pos[0]);
        } else {
            sol[0] = Kokkos::sin(k*pos[1])*Kokkos::sin(k*pos[2]);
            sol[1] = Kokkos::sin(k*pos[2])*Kokkos::sin(k*pos[0]);
            sol[2] = Kokkos::sin(k*pos[0])*Kokkos::sin(k*pos[1]);
        }
        return sol;
    }
};

template <typename T, unsigned Dim>
struct Rhs{
    using point_t =  ippl::Vector<T, Dim>;

    T k;
    Rhs(T k) : k(k) {}

    KOKKOS_FUNCTION const point_t operator() (const point_t& pos) const{
        point_t sol(0);
        if constexpr (Dim == 2) {
            sol[0] = (1. + k*k)*Kokkos::sin(k*pos[1]);
            sol[1] = (1. + k*k)*Kokkos::sin(k*pos[0]);
        }
        if constexpr (Dim == 3) {
            sol[0] = (1. + 2*k*k)*Kokkos::sin(k*pos[1])*Kokkos::sin(k*pos[2]);
            sol[1] = (1. + 2*k*k)*Kokkos::sin(k*pos[2])*Kokkos::sin(k*pos[0]);
            sol[2] = (1. + 2*k*k)*Kokkos::sin(k*pos[0])*Kokkos::sin(k*pos[1]);
        }
        return sol;
    }
};


template <typename T, unsigned Dim>
void testFEMSolver(const unsigned& numNodesPerDim, const T& domain_start = 0.0,
                   const T& domain_end = 1.0) {
    // start the timer
    static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("initTest");
    IpplTimings::startTimer(initTimer);

    Inform m("");
    Inform msg2all("", INFORM_ALL_NODES);

    using Mesh_t   = ippl::UniformCartesian<T, Dim>;
    using Field_t  = ippl::Field<ippl::Vector<T,Dim>, Dim, Mesh_t, Cell>;
    using BConds_t = ippl::BConds<Field_t, Dim>;
    using point_t =  ippl::Vector<T, Dim>;

    const unsigned numCellsPerDim = numNodesPerDim - 1;
    const unsigned numGhosts      = 1;

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
    Field_t lhs(mesh, layout, numGhosts);  // left hand side (updated in the algorithm)
    Field_t rhs(mesh, layout, numGhosts);  // right hand side (set once)

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<Field_t>>(i);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);

    // set k and retrieve the functors for the rhs and the solution.
    T k = 3.14159265359;
    Analytical<T,Dim> analytical(k);
    Rhs<T,Dim> rhsFunc(k);


    IpplTimings::stopTimer(initTimer);

    // initialize the solver
    ippl::FEMMaxwellDiffusionSolver<Field_t> solver(lhs, rhs, rhsFunc);

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 1e-13);
    params.add("max_iterations", 10000);
    solver.mergeParameters(params);
    
    // solve the problem
    ippl::FEMVector<ippl::Vector<T,Dim> > result = solver.solve();

    // Calculate error
    T coefError = solver.getL2ErrorCoeff(analytical);

    if (ippl::Comm->rank() == 0) {
        std::cout << std::setw(10) << numNodesPerDim;
        std::cout << std::setw(25) << std::setprecision(16) << cellSpacing[0];
        std::cout << std::setw(25) << std::setprecision(16) << coefError;
        std::cout << std::setw(25) << std::setprecision(16) << solver.getResidue();
        std::cout << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("");

        using T = double;

        unsigned dim = 3;

        if (argc > 1 && std::atoi(argv[1]) == 1) {
            dim = 1;
        } else if (argc > 1 && std::atoi(argv[1]) == 2) {
            dim = 2;
        }

        // start the timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        msg << std::setw(10) << "Size";
        msg << std::setw(25) << "Spacing";
        msg << std::setw(25) << "Relative Error";
        msg << std::setw(25) << "Residue";
        msg << std::setw(15) << "Iterations";
        msg << endl;

        if (ippl::Comm->rank() == 0) {
            std::cout << std::setw(10) << "num_nodes"
                      << std::setw(25) << "cell_spacing"
                      //<< std::setw(25) << "value_error"
                      //<< std::setw(25) << "interp_error"
                      << std::setw(25) << "interp_error_coef"
                      << std::setw(25) << "solver_residue"
                      << std::setw(15) << "num_it\n";
        }
        
        if (dim == 2) {
            // 2D Sinusoidal
            for (unsigned n = 32; n <= 256; n += 1) {
                testFEMSolver<T, 2>(n, 1.0, 3.0);
            }
        } else {
            // 3D Sinusoidal
            for (unsigned n = 32; n <= 256; n += 1) {
                testFEMSolver<T, 3>(n, 1.0, 3.0);
            }
        }

        // stop the timer
        IpplTimings::stopTimer(allTimer);

        // print the timers
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
