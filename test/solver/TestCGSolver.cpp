// Tests the conjugate gradient solver for Poisson problems
// by checking the relative error from the exact solution
// Usage:
//      TestCGSolver [size [scaling_type , preconditioner]]
//      ./TestCGSolver 6 j --info 5

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

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, 3>;
        using Centering_t          = Mesh_t::DefaultCentering;

        int pt = 4, ptY = 4;
        bool isWeak = false;
        // Preconditioner Setup Start
        int gauss_seidel_inner_iterations;
        int gauss_seidel_outer_iterations;
        int newton_level;
        int chebyshev_degree;
        int richardson_iterations;
        int communication;
        std::string solver              = "not preconditioned";
        std::string preconditioner_type = "";
        // Preconditioner Setup End
        Inform info("Config");
        if (argc >= 2) {
            // First argument is the problem size (log2)
            double N = strtol(argv[1], NULL, 10);
            info << "Got " << N << " as size parameter" << endl;
            pt = ptY = 1 << (int)N;
            if (argc >= 3) {
                if (argv[2][0] == 'w') {
                    // If weak scaling is specified, increase the problem size
                    // along the Y axis such that each rank has the same workload
                    // (the simplest enlargement method)
                    ptY = 1 << (5 + (int)N);
                    pt  = 32;
                    info << "Performing weak scaling" << endl;
                    isWeak = true;
                } else {
                    if (argv[2][0] == 'j') {
                        solver              = "preconditioned";
                        preconditioner_type = "jacobi";
                    }
                    if (argv[2][0] == 'n') {
                        solver              = "preconditioned";
                        preconditioner_type = "newton";
                        newton_level        = std::atoi(argv[3]);
                    }
                    if (argv[2][0] == 'c') {
                        solver              = "preconditioned";
                        preconditioner_type = "chebyshev";
                        chebyshev_degree    = std::atoi(argv[3]);
                    }
                    if (argv[2][0] == 'g') {
                        solver                        = "preconditioned";
                        preconditioner_type           = "gauss-seidel";
                        gauss_seidel_inner_iterations = std::atoi(argv[3]);
                        gauss_seidel_outer_iterations = std::atoi(argv[4]);
                        communication                 = std::atoi(argv[5]);
                    }
                    if (argv[2][0] == 'r') {
                        solver                = "preconditioned";
                        preconditioner_type   = "richardson";
                        richardson_iterations = std::atoi(argv[3]);
                        communication         = std::atoi(argv[4]);
                    }
                }
                if (argc >= 4) {
                    if (argv[3][0] == 'j') {
                        solver              = "preconditioned";
                        preconditioner_type = "jacobi";
                    }
                    if (argv[3][0] == 'n') {
                        solver              = "preconditioned";
                        preconditioner_type = "newton";
                        newton_level        = std::atoi(argv[4]);
                    }
                    if (argv[3][0] == 'c') {
                        solver              = "preconditioned";
                        preconditioner_type = "chebyshev";
                        chebyshev_degree    = std::atoi(argv[4]);
                    }
                    if (argv[3][0] == 'g') {
                        solver                        = "preconditioned";
                        preconditioner_type           = "gauss-seidel";
                        gauss_seidel_inner_iterations = std::atoi(argv[4]);
                        gauss_seidel_outer_iterations = std::atoi(argv[5]);
                        communication                 = std::atoi(argv[6]);
                    }
                    if (argv[3][0] == 'r') {
                        solver                = "preconditioned";
                        preconditioner_type   = "richardson";
                        richardson_iterations = std::atoi(argv[4]);
                        communication         = std::atoi(argv[5]);
                    }
                }
            }
        }
        info << "Solver is " << solver << endl;
        if (solver == "preconditioned") {
            info << "Preconditioner is " << preconditioner_type << endl;
        }

        ippl::Index I(pt), Iy(ptY);
        ippl::NDIndex<dim> owned(I, Iy, I);

        std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++) {
            isParallel[d] = true;
        }

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        // Unit box
        double dx                        = 2.0 / double(pt);
        double dy                        = 2.0 / double(ptY);
        ippl::Vector<double, dim> hx     = {dx, dy, dx};
        ippl::Vector<double, dim> origin = -1;
        Mesh_t mesh(owned, hx, origin);

        double pi = Kokkos::numbers::pi_v<double>;

        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;
        field_type rhs(mesh, layout), lhs(mesh, layout), solution(mesh, layout);

        typedef ippl::BConds<field_type, dim> bc_type;

        bc_type bcField;

        for (unsigned int i = 0; i < 6; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
        }

        lhs.setFieldBC(bcField);

        typename field_type::view_type &viewRHS = rhs.getView(), viewSol = solution.getView();

        const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();

        using Kokkos::pow, Kokkos::sin, Kokkos::cos;

        int shift1     = solution.getNghost();
        auto policySol = solution.getFieldRangePolicy();
        Kokkos::parallel_for(
            "Assign solution", policySol, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const size_t ig = i + lDom[0].first() - shift1;
                const size_t jg = j + lDom[1].first() - shift1;
                const size_t kg = k + lDom[2].first() - shift1;
                double x        = (ig + 0.5) * hx[0];
                double y        = (jg + 0.5) * hx[1];
                double z        = (kg + 0.5) * hx[2];

                viewSol(i, j, k) = sin(sin(pi * x)) * sin(sin(pi * y)) * sin(sin(pi * z));
            });

        const int shift2 = rhs.getNghost();
        auto policyRHS   = rhs.getFieldRangePolicy();
        Kokkos::parallel_for(
            "Assign rhs", policyRHS, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const size_t ig = i + lDom[0].first() - shift2;
                const size_t jg = j + lDom[1].first() - shift2;
                const size_t kg = k + lDom[2].first() - shift2;
                double x        = (ig + 0.5) * hx[0];
                double y        = (jg + 0.5) * hx[1];
                double z        = (kg + 0.5) * hx[2];

                // https://gitlab.psi.ch/OPAL/Libraries/ippl-solvers/-/blob/5-fftperiodicpoissonsolver/test/TestFFTPeriodicPoissonSolver.cpp#L91
                viewRHS(i, j, k) =
                    pow(pi, 2)
                    * (cos(sin(pi * z)) * sin(pi * z) * sin(sin(pi * x)) * sin(sin(pi * y))
                       + (cos(sin(pi * y)) * sin(pi * y) * sin(sin(pi * x))
                          + (cos(sin(pi * x)) * sin(pi * x)
                             + (pow(cos(pi * x), 2) + pow(cos(pi * y), 2) + pow(cos(pi * z), 2))
                                   * sin(sin(pi * x)))
                                * sin(sin(pi * y)))
                             * sin(sin(pi * z)));
            });

        ippl::PoissonCG<field_type> lapsolver;

        ippl::ParameterList params;
        params.add("max_iterations", 2000);
        params.add("solver", solver);
        // Preconditioner Setup
        params.add("preconditioner_type", preconditioner_type);
        params.add("gauss_seidel_inner_iterations", gauss_seidel_inner_iterations);
        params.add("gauss_seidel_outer_iterations", gauss_seidel_outer_iterations);
        params.add("newton_level", newton_level);
        params.add("chebyshev_degree", chebyshev_degree);
        params.add("richardson_iterations", richardson_iterations);
        params.add("communication", communication);

        lapsolver.mergeParameters(params);

        lapsolver.setRhs(rhs);
        lapsolver.setLhs(lhs);

        lhs = 0;
        lapsolver.solve();

        const char* name = isWeak ? "Convergence (weak)" : "Convergence";
        Inform m(name);

        field_type error(mesh, layout);
        // Solver solution - analytical solution
        error           = lhs - solution;
        double relError = norm(error) / norm(solution);

        // Laplace(solver solution) - rhs
        error          = -laplace(lhs) - rhs;
        double residue = norm(error) / norm(rhs);

        int size    = isWeak ? pt * pt * ptY : pt;
        int itCount = lapsolver.getIterationCount();
        m << size << "," << std::setprecision(16) << relError << "," << residue << "," << itCount
          << endl;

        IpplTimings::stopTimer(allTimer);
        IpplTimings::print("timings" + std::to_string(pt) + ".dat");
    }
    ippl::finalize();

    return 0;
}
