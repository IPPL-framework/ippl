// Periodic BCs scaling test
//
// Run the solve with periodic BCs 5 times,
// with problem size as user input.
// This is the 3D case.
// THIS IS WITH PRECONDITIONING.
//
// Uses the sinusoidal as source function
// (see TestPeriodicBC_sinsin.cpp)
//
// Usage:
//  ./TestScaling_PeriodicBC_sinsin_precon <pt> --info 5

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <typeinfo>

#include "Utility/Inform.h"
#include "Utility/IpplTimings.h"

#include "PoissonSolvers/PoissonCG.h"
#include "PoissonSolvers/PreconditionedFEMPoissonSolver.h"

template <typename T, unsigned Dim>
KOKKOS_INLINE_FUNCTION T sinusoidalRHSFunction(ippl::Vector<T, Dim> x_vec) {
    const T pi = Kokkos::numbers::pi_v<T>;

    T val = 1.0;
    if (Dim == 1) {
        T x = x_vec[0];

        val = Kokkos::pow(pi, 2) * ((Kokkos::cos(Kokkos::sin(pi * x)) * Kokkos::sin(pi * x))
                + (Kokkos::pow(Kokkos::cos(pi * x), 2) * Kokkos::sin(Kokkos::sin(pi * x))));

    } else if (Dim == 2) {
        T x = x_vec[0];
        T y = x_vec[1];

        val = Kokkos::pow(pi, 2)
                * (Kokkos::cos(Kokkos::sin(pi * y)) * Kokkos::sin(pi * y) * Kokkos::sin(Kokkos::sin(pi * x))
                + (Kokkos::cos(Kokkos::sin(pi * x)) * Kokkos::sin(pi * x)
                + (Kokkos::pow(Kokkos::cos(pi * x), 2) + Kokkos::pow(Kokkos::cos(pi * y), 2)) * Kokkos::sin(Kokkos::sin(pi * x)))
                * Kokkos::sin(Kokkos::sin(pi * y)));
                
    } else if (Dim == 3) {
        T x = x_vec[0];
        T y = x_vec[1];
        T z = x_vec[2];

        val = Kokkos::pow(pi, 2)
                * (Kokkos::cos(Kokkos::sin(pi * z)) * Kokkos::sin(pi * z) * Kokkos::sin(Kokkos::sin(pi * x)) * Kokkos::sin(Kokkos::sin(pi * y))
                + (Kokkos::cos(Kokkos::sin(pi * y)) * Kokkos::sin(pi * y) * Kokkos::sin(Kokkos::sin(pi * x))
                + (Kokkos::cos(Kokkos::sin(pi * x)) * Kokkos::sin(pi * x)
                + (Kokkos::pow(Kokkos::cos(pi * x), 2) + Kokkos::pow(Kokkos::cos(pi * y), 2) + Kokkos::pow(Kokkos::cos(pi * z), 2))
                * Kokkos::sin(Kokkos::sin(pi * x)))
                * Kokkos::sin(Kokkos::sin(pi * y)))
                * Kokkos::sin(Kokkos::sin(pi * z)));
    }

    return val;
}

template <typename T, unsigned Dim>
struct AnalyticSol {
    const T pi = Kokkos::numbers::pi_v<T>;

    KOKKOS_FUNCTION const T operator()(ippl::Vector<T, Dim> x_vec) const {
        T val = 1.0;
        for (unsigned d = 0; d < Dim; d++) {
            val *= Kokkos::sin(Kokkos::sin(pi * x_vec[d]));
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

        // start the timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        msg << std::setw(10) << "Size";
        msg << std::setw(25) << "Spacing";
        msg << std::setw(25) << "Relative Error";
        msg << std::setw(25) << "Residue";
        msg << std::setw(15) << "Iterations";
        msg << endl;

        unsigned pt = std::atoi(argv[1]);

        for (unsigned n = 0; n < 5; ++n) {
            testFEMSolver<T, 3>(pt, -1.0, 1.0);
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
