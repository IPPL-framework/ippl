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

template <typename T>
KOKKOS_INLINE_FUNCTION T piSquaredSinPiX1D(T x) {
    const T pi = Kokkos::numbers::pi_v<T>;

    return pi * pi * Kokkos::sin(pi * x);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T sinPiX1D(T x) {
    const T pi = Kokkos::numbers::pi_v<T>;

    return Kokkos::sin(pi * x);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussian(T x, T y, T z, T sigma = 0.05, T mu = 0.5) {
    const T pi = Kokkos::numbers::pi_v<T>;

    const T prefactor =
        (1 / Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    const T r2 = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return prefactor * exp(-r2 / (2 * sigma * sigma));
}

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussianSol(T x, T y, T z, T sigma = 0.05, T mu = 0.5) {
    const T pi = Kokkos::numbers::pi_v<T>;

    const T r = Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));

    return (1 / (4.0 * pi * r)) * Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma));
}

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussian1D(T x) {
    const T sigma = 0.05;
    const T mu    = 0.5;

    const T pi = Kokkos::numbers::pi_v<T>;

    const T prefactor = (1 / Kokkos::sqrt(2 * pi)) * (1 / sigma);
    const T r2        = (x - mu) * (x - mu);

    return prefactor * exp(-r2 / (2 * sigma * sigma));
}

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussianSol1D(T x) {
    const T sigma = 0.05;
    const T mu    = 0.5;

    const T pi = Kokkos::numbers::pi_v<T>;

    const T r = Kokkos::sqrt((x - mu) * (x - mu));

    return (1.0 / 4.0 * pi * r) * Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma));
}

/**
 * Test problem in 1D:
 *
 * -Laplacian(u) = pi^2 * sin(pi * x), x in [-1,1]
 * u(-1) = u(1) = 0
 *
 * Exact solution is u(x) = sin(pi * x)
 */
template <typename T>
void testFEMSolver(const unsigned numNodesPerDim, std::function<T(T x)> f_rhs,
                   std::function<T(T x)> f_sol) {
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
    Kokkos::parallel_for(
        "Assign solution", sol.getFieldRangePolicy(), KOKKOS_LAMBDA(const int i) {
            const T x = (i - numGhosts) * cellSpacing[0] + origin[0];

            sol.getView()(i) = f_sol(x);
        });

    auto f = [&f_rhs](ippl::Vector<T, 1> x) {
        return f_rhs(x[0]);
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

    // DEBUG PRINT OF LHS
    std::cout << std::setw(15) << "LHS: ";
    for (unsigned int i = 0; i < lhs.getView().size(); ++i) {
        std::cout << std::setw(15) << lhs(i);
    }
    std::cout << std::endl;

    std::cout << std::setw(15) << "SOL: ";
    for (unsigned int i = 0; i < sol.getView().size(); ++i) {
        std::cout << std::setw(15) << sol(i);
    }
    std::cout << std::endl;

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
        using T = double;

        Inform msg("");

        // start the timer
        static IpplTimings::TimerRef timer = IpplTimings::getTimer("timer");
        IpplTimings::startTimer(timer);

        msg << std::setw(10) << "Size";
        msg << std::setw(25) << "Relative Error";
        msg << std::setw(25) << "Residue";
        msg << std::setw(15) << "Iterations";
        msg << endl;

        const std::vector<T> N = {4, 8};  //, 16, 32, 64, 128};

        for (const T numPoints : N) {
            // testFEMSolver<T>(numPoints, piSquaredSinPiX1D<T>, sinPiX1D<T>);
            testFEMSolver<T>(numPoints, gaussian1D<T>, gaussianSol1D<T>);
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