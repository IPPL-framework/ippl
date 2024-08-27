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

/*
template <typename T, unsigned Dim>
KOKKOS_INLINE_FUNCTION T sinusoidalRHSFunction(ippl::Vector<T, Dim> x_vec) {
    const T pi = Kokkos::numbers::pi_v<T>;

    T val = 1.0;
    for (unsigned d = 0; d < Dim; d++) {
        val *= Kokkos::sin(pi * x_vec[d]);
    }

    return Dim * pi * pi * val;
}
*/

template <typename T, unsigned Dim>
KOKKOS_INLINE_FUNCTION T sinusoidalSolution(ippl::Vector<T, Dim> x_vec) {
    const T pi = Kokkos::numbers::pi_v<T>;

    T val = 1.0;
    for (unsigned d = 0; d < Dim; d++) {
        val *= Kokkos::sin(pi * x_vec[d]);
    }
    return val;
}

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussian3d(ippl::Vector<T, 3> x_vec) {
    const T& x = x_vec[0];
    const T& y = x_vec[1];
    const T& z = x_vec[2];

    const T sigma = 0.05;
    const T mu    = 0.5;

    const T pi = Kokkos::numbers::pi_v<T>;

    const T prefactor =
        (1.0 / Kokkos::sqrt(2.0 * 2.0 * 2.0 * pi * pi * pi)) * (1.0 / (sigma * sigma * sigma));
    const T r2 = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

    return prefactor * Kokkos::exp(-r2 / (2.0 * sigma * sigma));
}

template <typename T>
KOKKOS_INLINE_FUNCTION T gaussian3dSol(ippl::Vector<T, 3> x_vec) {
    const T& x = x_vec[0];
    const T& y = x_vec[1];
    const T& z = x_vec[2];

    const T sigma = 0.05;
    const T mu    = 0.5;

    const T pi = Kokkos::numbers::pi_v<T>;

    const T r = Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));

    return (1.0 / (4.0 * pi * r)) * Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma));
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
void testFEMSolver(const unsigned& numNodesPerDim,
                   const T& domain_start = 0.0,
                   const T& domain_end = 1.0) {
                   // std::function<T(ippl::Vector<T, Dim> x)> f_sol,
    // start the timer
    static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("initTest");
    IpplTimings::startTimer(initTimer);

    Inform m("");
    Inform msg2all("", INFORM_ALL_NODES);

    m << "inside test fem" << endl;

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
    Field_t sol(mesh, layout, numGhosts);  // exact solution

    // Define boundary conditions
    BConds_t bcField;
    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<Field_t>>(i);
    }
    lhs.setFieldBC(bcField);
    rhs.setFieldBC(bcField);

    // set solution
    auto view = sol.getView();
    auto ldom = layout.getLocalNDIndex();

    m << "before assigning solution" << endl;

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    ippl::parallel_for("Assign solution", sol.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const index_array_type& args) {
            ippl::Vector<int, Dim> iVec = args - numGhosts;
            for (unsigned d = 0; d < Dim; ++d) {
                iVec[d] += ldom[d].first();
            }
            const ippl::Vector<T, Dim> x = (iVec * cellSpacing) + origin;
            
            apply(view, args) = sinusoidalSolution<T, Dim>(x);
        });

    IpplTimings::stopTimer(initTimer);

    Kokkos::fence();
    sol.write();
    Kokkos::fence();

    m << "after assigning solution" << endl;

    // initialize the solver
    ippl::FEMPoissonSolver<Field_t, Field_t> solver(lhs, rhs);

    // set the parameters
    ippl::ParameterList params;
    params.add("tolerance", 1e-13);
    params.add("max_iterations", 2000);
    solver.mergeParameters(params);

    // solve the problem
    solver.solve();

    // start the timer
    static IpplTimings::TimerRef errorTimer = IpplTimings::getTimer("computeError");
    IpplTimings::startTimer(errorTimer);

    // Compute the error
    Field_t error(mesh, layout, numGhosts);
    error                 = lhs - sol;
    const double relError = norm(error) / norm(sol);

    m << std::setw(10) << numNodesPerDim;
    m << std::setw(25) << std::setprecision(16) << cellSpacing[0];
    m << std::setw(25) << std::setprecision(16) << relError;
    m << std::setw(25) << std::setprecision(16) << solver.getResidue();
    m << std::setw(15) << std::setprecision(16) << solver.getIterationCount();
    m << endl;

    IpplTimings::stopTimer(errorTimer);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

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

        if (dim == 1) {
            // 1D Sinusoidal
            for (unsigned n = 1 << 2; n <= 1 << 2; n = n << 1) {
                /*testFEMSolver<T, 1>(n, 
                    [](ippl::Vector<T, 1> x) {
                        return gaussian1D<T>(x[0], 0.05, 0.5);
                    },
                    [](ippl::Vector<T, 1> x) {
                        return gaussianSol1D<T>(x[0], 0.05, 0.5);
                    },
                    0.0, 1.0);
                */
                testFEMSolver<T, 1>(n, -1.0, 1.0);
            }
        } else if (dim == 2) {
            // 2D Sinusoidal
            for (unsigned n = 1 << 3; n <= 1 << 10; n = n << 1) {
                //testFEMSolver<T, 2>(n, sinusoidalRHSFunction<T, 2>, sinusoidalSolution<T, 2>, -1.0, 1.0);
                testFEMSolver<T, 2>(n, -1.0, 1.0);
            }
        } else {
            // 3D Sinusoidal; problem size given by user
            const int n_arg = std::atoi(argv[1]);
            int n = 1 << n_arg;
            //testFEMSolver<T, 3>(n, gaussian3d<T>, gaussian3dSol<T>, 0.0, 1.0);
            testFEMSolver<T, 3>(n, -1.0, 1.0);
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
