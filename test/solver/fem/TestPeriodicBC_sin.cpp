// Tests the FEM Poisson solver by solving the problem:
//
// -Laplacian(u) = pi^2 * sin(pi * x), x in [-1,1]
// u(-1) = u(1) = 0
//
// Exact solution is u(x) = sin(pi * x).
//
// The test prints out the relative error as we refine
// the mesh spacing i.e. it is a convergence study. 
// The order of convergence should be 2. 
//
// The test is available in 1D (problem above),
// as well as 2D and 3D with analogous test cases.
//
// Here we use periodic BCs, so this should work
// for other domains too as long as the domain 
// length is the size of the period i.e. 2.
//
// Usage:
//     ./TestPeriodicBC_sin <dim> --info 5

#include "Ippl.h"

#include "Meshes/Centering.h"
#include "PoissonSolvers/FEMPoissonSolver.h"

template <typename T, unsigned Dim>
KOKKOS_INLINE_FUNCTION T sinusoidalRHSFunction(ippl::Vector<T, Dim> x_vec) {
    const T pi = Kokkos::numbers::pi_v<T>;

    T val = 1.0;
    for (unsigned d = 0; d < Dim; d++) {
        val *= Kokkos::sin(pi * x_vec[d]);
    }

    return Dim * pi * pi * val;
}

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

template <typename T, unsigned Dim>
void testFEMSolver(const unsigned& numNodesPerDim, const T& domain_start = 0.0,
                   const T& domain_end = 1.0) {
    // start the timer
    static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("initTest");
    IpplTimings::startTimer(initTimer);

    Inform m("");
    Inform msg2all("", INFORM_ALL_NODES);

    using Mesh_t   = ippl::UniformCartesian<T, Dim>;
    using Field_t  = ippl::Field<T, Dim, Mesh_t, Cell>;
    using BConds_t = ippl::BConds<Field_t, Dim>;

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
    Field_t rhs(mesh, layout, numGhosts);  // left hand side (updated in the algorithm)

    Field_t analytical(mesh, layout, numGhosts);  // right hand side (set once)
    auto view_analytical = analytical.getView();

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::PeriodicFace<Field_t>>(i);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);

    // set rhs
    auto view_rhs = rhs.getView();
    auto ldom     = layout.getLocalNDIndex();

    AnalyticSol<T, Dim> analytic;

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for(
        "Assign RHS", rhs.getFieldRangePolicy(), KOKKOS_LAMBDA(const index_array_type& args) {
            ippl::Vector<int, Dim> iVec = args - numGhosts;
            for (unsigned d = 0; d < Dim; ++d) {
                iVec[d] += ldom[d].first();
            }

            const ippl::Vector<T, Dim> x = (iVec)*cellSpacing + origin;

            apply(view_rhs, args) = sinusoidalRHSFunction<T, Dim>(x);
            apply(view_analytical, args) = analytic(x);
        });

    IpplTimings::stopTimer(initTimer);

    // initialize the solver
    ippl::FEMPoissonSolver<Field_t, Field_t> solver(lhs, rhs);

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 1e-13);
    params.add("max_iterations", 2000);
    solver.mergeParameters(params);

    // solve the problem
    solver.solve();

    // average to 0 since constant null space (there can be any additive constant)
    lhs = lhs - solver.getAvg(true);

    // start the timer
    static IpplTimings::TimerRef errorTimer = IpplTimings::getTimer("computeError");
    IpplTimings::startTimer(errorTimer);

    // Compute the error
    const T relError = solver.getL2Error(analytic);

    lhs = lhs - analytical;
    T normError = norm(lhs) / norm(analytical);

    m << std::setw(10) << numNodesPerDim;
    m << std::setw(25) << std::setprecision(16) << cellSpacing[0];
    m << std::setw(25) << std::setprecision(16) << relError;
    m << std::setw(25) << std::setprecision(16) << normError;
    m << std::setw(25) << std::setprecision(16) << solver.getResidue();
    m << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
    m << endl;

    IpplTimings::stopTimer(errorTimer);
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
        msg << std::setw(25) << "Norm Error";
        msg << std::setw(25) << "Residue";
        msg << std::setw(15) << "Iterations";
        msg << endl;

        if (dim == 1) {
            // 1D Sinusoidal
            for (unsigned n = 1 << 3; n <= 1 << 10; n = n << 1) {
                testFEMSolver<T, 1>(n, 0.0, 2.0);
            }
        } else if (dim == 2) {
            // 2D Sinusoidal
            for (unsigned n = 1 << 3; n <= 1 << 10; n = n << 1) {
                testFEMSolver<T, 2>(n, 0.0, 2.0);
            }
        } else {
            // 3D Sinusoidal
            for (unsigned n = 1 << 3; n <= 1 << 9; n = n << 1) {
                testFEMSolver<T, 3>(n, 0.0, 2.0);
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
