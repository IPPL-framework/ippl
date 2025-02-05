// Tests the FEM Poisson solver with a Gaussian source
// and homogeneous Dirichlet boundaries
//
// Usage:
//     ./TestNonHomDirichlet --info 5

#include "Ippl.h"

#include "Meshes/Centering.h"
#include "PoissonSolvers/FEMPoissonSolver.h"

template <typename T>
struct AnalyticSol {
    KOKKOS_FUNCTION const T operator()(ippl::Vector<T, 1> x_vec) const {
        return (1.0 + 0.5*x_vec[0] - 0.5*(x_vec[0]*x_vec[0]));
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
    Field_t rhs(mesh, layout, numGhosts);  // right hand side (set once)
    Field_t sol(mesh, layout, numGhosts);  // right hand side (set once)

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<Field_t>>(i);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);
    bcField.apply(lhs);
    bcField.apply(rhs);

    //std::cout << "write lhs" << std::endl;
    //lhs.write();
    
    rhs = 1.0;

    //std::cout << "rhs set = " << std::endl;
    //rhs.write();

    // set rhs
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

            //std::cout << "x = " << x << std::endl;

            apply(view_sol, args) = analytic(x);
        });

    IpplTimings::stopTimer(initTimer);

    // initialize the solver
    const T dirichlet = 1.0;
    ippl::FEMPoissonSolver<Field_t, Field_t> solver(lhs, rhs, dirichlet);

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 1e-7);
    params.add("max_iterations", 100);

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

        for (unsigned n = 1 << 2; n <= 1 << 3; n = n << 1) {
            testFEMSolver<T, 1>(n, 0.0, 1.0);
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
