// Tests the FEM Poisson solver WITH PRECONDITIONING
// by solving the problem:
//
// -Laplacian(u) = f(x, y),
// where x,y in [0,1]^2 and u(boundaries) = 1.56, 
// and f(x,y) is such that the exact solution is 
// u(x,y) = x^2(1 - x^2) + y^2(1 - y^2) + 1.56.
//
// BCs: Dirichlet BCs (Constant Face = 1.56).
// This is only 2D!
//
// The test prints out the relative error as we refine
// the mesh spacing i.e. it is a convergence study. 
// The order of convergence should be 2. 
//
// Usage:
//     ./TestNonHomDirichlet_2d_preconditioned --info 5

#include "Ippl.h"

#include "Meshes/Centering.h"
#include "PoissonSolvers/PreconditionedFEMPoissonSolver.h"

template <typename T>
struct AnalyticSol {
    KOKKOS_FUNCTION const T operator()(ippl::Vector<T, 2> x_vec) const {
        return (x_vec[0] * x_vec[0]) * (1.0 - x_vec[0] * x_vec[0])
               * (x_vec[1] * x_vec[1]) * (1.0 - x_vec[1]*x_vec[1]) + 1.56;
    }
};

template <typename T>
KOKKOS_INLINE_FUNCTION T rhs_function(ippl::Vector<T, 2> x_vec) {
    double x2 = x_vec[0] * x_vec[0];
    double y2 = x_vec[1] * x_vec[1];
    double x4 = Kokkos::pow(x_vec[0], 4);
    double y4 = Kokkos::pow(x_vec[1], 4);

    return -2.0*(y2 - y4 + (x4*(-1.0 + 6*y2)) + x2*(1.0-12*y2+6*y4));
}

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

    // Domain: [0, 1]
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
    Field_t sol(mesh, layout, numGhosts);  // right hand side (set once)

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ConstantFace<Field_t>>(i, 1.56);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);
    bcField.apply(lhs);
    bcField.apply(rhs);

    // set analytic sol and rhs
    auto view_rhs = rhs.getView();
    auto view_sol = sol.getView();
    auto ldom     = layout.getLocalNDIndex();

    AnalyticSol<T> analytic;

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for(
        "Assign RHS", rhs.getFieldRangePolicy(), KOKKOS_LAMBDA(const index_array_type& args) {
            ippl::Vector<int, Dim> iVec = args - numGhosts;
            for (unsigned d = 0; d < Dim; ++d) {
                iVec[d] += ldom[d].first();
            }
            const ippl::Vector<T, Dim> x = (iVec)*cellSpacing + origin;

            apply(view_rhs, args) = rhs_function<T>(x);
            apply(view_sol, args) = analytic(x);
        });

    IpplTimings::stopTimer(initTimer);

    // initialize the solver
    ippl::PreconditionedFEMPoissonSolver<Field_t, Field_t> solver(lhs, rhs);

    // parameters for the preconditioner
    std::string preconditioner_type = "richardson";
    int gauss_seidel_inner_iterations = 4;
    int gauss_seidel_outer_iterations = 2;
    int newton_level = 1; // unused
    int chebyshev_degree = 1; // unused
    int richardson_iterations = 4;
    double ssor_omega = 1.57079632679;

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 1e-13);
    params.add("max_iterations", 2000);
    // preconditioner params
    params.add("preconditioner_type", preconditioner_type);
    params.add("gauss_seidel_inner_iterations", gauss_seidel_inner_iterations);
    params.add("gauss_seidel_outer_iterations", gauss_seidel_outer_iterations);
    params.add("newton_level", newton_level);
    params.add("chebyshev_degree", chebyshev_degree);
    params.add("richardson_iterations", richardson_iterations);
    params.add("ssor_omega", ssor_omega);
    solver.mergeParameters(params);

    // solve the problem
    solver.solve();

    // start the timer
    static IpplTimings::TimerRef errorTimer = IpplTimings::getTimer("computeError");
    IpplTimings::startTimer(errorTimer);

    // Compute the error
    const T relError = solver.getL2Error(analytic);

    lhs = lhs - sol;
    const T normError = norm(lhs) / norm(sol);

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

        for (unsigned n = 1 << 3; n <= 1 << 8; n = n << 1) {
            testFEMSolver<T, 2>(n, 0.0, 1.0);
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
