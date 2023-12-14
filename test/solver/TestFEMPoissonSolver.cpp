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
KOKKOS_INLINE_FUNCTION T gaussian(ippl::Vector<T, 3> x_vec) {
    const T& x = x_vec[0];
    const T& y = x_vec[1];
    const T& z = x_vec[2];

    const T sigma = 0.05;
    const T mu    = 0.5;

    const T pi = Kokkos::numbers::pi_v<T>;

    const T prefactor =
        (1 / Kokkos::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
    const T r2 = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return prefactor * Kokkos::exp(-r2 / (2 * sigma * sigma));
}

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussianSol(ippl::Vector<T, 3> x_vec) {
    const T& x = x_vec[0];
    const T& y = x_vec[1];
    const T& z = x_vec[2];

    const T sigma = 0.05;
    const T mu    = 0.5;

    const T pi = Kokkos::numbers::pi_v<T>;

    const T r = Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));

    return (1 / (4.0 * pi * r)) * Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma));
}

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussian1D(const T& x, const T& sigma = 0.05, const T& mu = 0.5) {
    const T pi = Kokkos::numbers::pi_v<T>;

    const T prefactor = (1.0 / Kokkos::sqrt(2.0 * pi)) * (1.0 / sigma);
    const T r2        = (x - mu) * (x - mu);

    return prefactor * Kokkos::exp(-r2 / (2.0 * sigma * sigma));
}

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussianSol1D(const T& x, const T& sigma = 0.05, const T& mu = 0.5) {
    const T pi     = Kokkos::numbers::pi_v<T>;
    const T sqrt_2 = Kokkos::sqrt(2.0);

    const T r = (x - mu);

    return (-sigma / sqrt_2)
               * ((r / (sqrt_2 * sigma)) * Kokkos::erf(r / (sqrt_2 * sigma))
                  + (1.0 / Kokkos::sqrt(pi)) * Kokkos::exp(-(r * r) / (2.0 * sigma * sigma)))
           + (0.5 - mu) * x + 0.5 * mu;
}

template <typename T, unsigned Dim>
void testFEMSolver(const unsigned& numNodesPerDim, std::function<T(ippl::Vector<T, Dim> x)> f_rhs,
                   std::function<T(ippl::Vector<T, Dim> x)> f_sol, const T& domain_start = 0.0,
                   const T& domain_end = 1.0) {
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

    ippl::FieldLayout<Dim> layout(domain);
    Field_t lhs(mesh, layout, numGhosts);  // left hand side (updated in the algorithm)
    Field_t rhs(mesh, layout, numGhosts);  // right hand side (set once)
    Field_t sol(mesh, layout, numGhosts);  // exact solution

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<Field_t>>(i);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);

    // set solution
    // if (dim == 1) {
    Kokkos::parallel_for(
        "Assign solution", sol.getFieldRangePolicy(), KOKKOS_LAMBDA(const unsigned i) {
            const ippl::Vector<unsigned, Dim> indices{i};
            const ippl::Vector<T, Dim> x = (indices - numGhosts) * cellSpacing + origin;

            sol.getView()(i) = f_sol(x);
        });
    // } else if (dim == 3) {
    //     Kokkos::parallel_for(
    //         "Assign solution", sol.getFieldRangePolicy(),
    //         KOKKOS_LAMBDA(const unsigned i, const unsigned j, const unsigned k) {
    //             const ippl::Vector<unsigned, Dim> indices{i, j, k};
    //             const ippl::Vector<T, Dim> x = (indices - numGhosts) * cellSpacing + origin;

    //             sol.getView()(i, j, k) = f_sol(x);
    //         });
    // }

    // initialize the solver
    ippl::FEMPoissonSolver<Field_t, Field_t> solver(lhs, rhs, f_rhs);

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

        const std::vector<T> N = {4, 8, 16, 32, 64, 128};

        for (const T numPoints : N) {
            // testFEMSolver<T, 1>(numPoints, piSquaredSinPiX1D<T>, sinPiX1D<T>, -1.0, 1.0);
            // testFEMSolver<T, 3>(numPoints, gaussian<T>, gaussianSol<T>, 0.0, 1.0);
            testFEMSolver<T, 1>(
                numPoints,
                [](ippl::Vector<T, 1> x) {
                    return gaussian1D<T>(x[0], 0.05, 0.5);
                },
                [](ippl::Vector<T, 1> x) {
                    return gaussianSol1D<T>(x[0], 0.05, 0.5);
                },
                0.0, 1.0);
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