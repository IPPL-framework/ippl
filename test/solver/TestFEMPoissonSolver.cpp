// Tests the FEM Poisson solver by solving the problem:
//
// -Laplacian(u) = pi^2 * sin(pi * x), x in [-1,1]
// u(-1) = u(1) = 0
//
// Exact solution is u(x) = sin(pi * x)
//

#include "Ippl.h"

#include "Meshes/Centering.h"
#include "PoissonSolvers/FEMPoissonSolver.h"

/**
 * Test problem in 1D:
 *
 * -Laplacian(u) = pi^2 * sin(pi * x), x in [-1,1]
 * u(-1) = u(1) = 0
 *
 * Exact solution is u(x) = sin(pi * x)
 */
template <typename T>
void testFEMSolver1D(const unsigned numNodesPerDim) {
    constexpr unsigned dim = 1;

    using Mesh_t   = ippl::UniformCartesian<T, dim>;
    using Field_t  = ippl::Field<T, dim, Mesh_t, Cell>;
    using BConds_t = ippl::BConds<Field_t, dim>;

    const unsigned numCellsPerDim = numNodesPerDim - 1;
    const unsigned numGhosts      = 1;

    // Domain: [-1, 1]
    ippl::NDIndex<dim> domain(numNodesPerDim);
    ippl::Vector<T, dim> cellSpacing(2.0 / static_cast<T>(numCellsPerDim));
    ippl::Vector<T, dim> origin(-1.0);
    Mesh_t mesh(domain, cellSpacing, origin);

    ippl::FieldLayout<dim> layout(domain);
    Field_t lhs(mesh, layout, numGhosts);  // left hand side (updated in the algorithm)
    Field_t rhs(mesh, layout, numGhosts);  // right hand side (set once)
    Field_t sol(mesh, layout, numGhosts);  // exact solution

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<Field_t>>(i);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);

    // set solution
    const T pi = Kokkos::numbers::pi_v<T>;
    Kokkos::parallel_for(
        "Assign solution", sol.getFieldRangePolicy(), KOKKOS_LAMBDA(const int i) {
            const T x = (i - numGhosts) * cellSpacing[0] + origin[0];

            sol.getView()(i) = Kokkos::sin(pi * x);
        });

    auto f = [&pi](ippl::Vector<T, 1> x) {
        return pi * pi * Kokkos::sin(pi * x[0]);
    };

    // initialize the solver
    ippl::FEMPoissonSolver<Field_t, Field_t> solver(lhs, rhs, f);

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 1e-15);
    params.add("max_iterations", 1000);
    solver.mergeParameters(params);

    // solve the problem
    solver.solve();

    Inform m("");

    // Compute the error
    Field_t error(mesh, layout, numGhosts);
    error                 = lhs - sol;
    const double relError = norm(error) / norm(sol);

    m << std::setw(10) << numNodesPerDim;
    m << std::setw(25) << std::setprecision(16) << relError;
    m << std::setw(25) << std::setprecision(16) << solver.getResidue();
    m << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
    m << endl;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("");

        // start the timer
        static IpplTimings::TimerRef timer = IpplTimings::getTimer("timer");
        IpplTimings::startTimer(timer);

        msg << std::setw(10) << "Size";
        msg << std::setw(25) << "Relative Error";
        msg << std::setw(25) << "Residue";
        msg << std::setw(15) << "Iterations";
        msg << endl;

        const std::array<unsigned, 6> N = {4, 8, 16, 32, 64, 128};

        for (const unsigned numPoints : N) {
            testFEMSolver1D<double>(numPoints);
        }

        // stop the timer
        IpplTimings::stopTimer(timer);

        // print
        IpplTimings::print();
        IpplTimings::print("fem_solver_timings.dat");
    }
    ippl::finalize();

    return 0;
}