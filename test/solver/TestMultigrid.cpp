// Tests the CG solver preconditioned with the Multigrid preconditioner
// by checking the relative error from the exact solution
//
// Usage:
//      ./TestMultigridCGSolver [log2_size] [test_case]
//
// Examples:
//      ./TestMultigrid 5 1   -> size 32^3, Test Case 1 (All Periodic)
//      ./TestMultigrid 6 2   -> size 64^3, Test Case 2 (Mixed Periodic/Dirichlet)

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>
#include <string>

#include "Utility/Inform.h"
#include "Utility/IpplTimings.h"

// Note: Ensure your PoissonCG solver is configured to recognize the
// "multigrid" parameter and instantiate your multigrid_preconditioner!
#include "PoissonSolvers/PoissonCG.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;
        using field_type           = ippl::Field<double, dim, Mesh_t, Centering_t>;
        using bc_type              = ippl::BConds<field_type, dim>;

        Inform info("MultigridTest");

        // Default parameters
        int pt        = 32;  // 32^3 grid
        int test_case = 1;   // 1 = Periodic, 2 = Mixed BCs

        if (argc >= 2) {
            double N = std::strtol(argv[1], NULL, 10);
            pt       = 1 << static_cast<int>(N);
        }
        if (argc >= 3) {
            test_case = std::atoi(argv[2]);
        }
        const char* tc_name = (test_case == 1)   ? " (All Periodic)"
                              : (test_case == 2) ? " (Mixed Periodic/Dirichlet)"
                                                 : " (All ConstantFace)";

        info << "Grid Size: " << pt << "^3" << endl;
        info << "Test Case: " << test_case << tc_name << endl;

        // 1. Setup Domain and Layout
        ippl::Index I(pt);
        ippl::NDIndex<dim> owned(I, I, I);
        std::array<bool, dim> isParallel = {true, true, true};
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        // 2. Setup Mesh, Spacing, and Boundary Conditions based on Test Case
        double dx, dy, dz;
        ippl::Vector<double, dim> origin = 0.0;
        bc_type bcField;

        if (test_case == 1) {
            dx = dy = dz = 2.0 / double(pt);
            for (unsigned int i = 0; i < 2 * dim; ++i) {
                bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
            }
        } else if (test_case == 2) {
            dx = 2.0 / double(pt);
            dy = 1.0 / double(pt);
            dz = 1.0 / double(pt);

            bcField[0] = std::make_shared<ippl::PeriodicFace<field_type>>(0);
            bcField[1] = std::make_shared<ippl::PeriodicFace<field_type>>(1);
            bcField[2] = std::make_shared<ippl::ZeroFace<field_type>>(2);
            bcField[3] = std::make_shared<ippl::ZeroFace<field_type>>(3);
            bcField[4] = std::make_shared<ippl::ZeroFace<field_type>>(4);
            bcField[5] = std::make_shared<ippl::ZeroFace<field_type>>(5);
        } else {
            // Test 3: All ConstantFace on [0,1]^3 with value bc_const
            dx = dy = dz = 1.0 / double(pt);
            for (unsigned int i = 0; i < 2 * dim; ++i) {
                bcField[i] = std::make_shared<ippl::ConstantFace<field_type>>(i, 0.0);
            }
        }

        ippl::Vector<double, dim> hx = {dx, dy, dz};
        Mesh_t mesh(owned, hx, origin);

        // 3. Initialize Fields
        field_type rhs(mesh, layout), lhs(mesh, layout), solution(mesh, layout);

        // Apply BCs to fields
        lhs.setFieldBC(bcField);
        rhs.setFieldBC(bcField);
        solution.setFieldBC(bcField);

        // 4. Fill RHS and Exact Solution
        const double pi                = Kokkos::numbers::pi_v<double>;
        const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();
        const int nghost               = solution.getNghost();

        auto viewSol = solution.getView();
        auto viewRHS = rhs.getView();

        Kokkos::parallel_for(
            "Assign exact solution and RHS", solution.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const size_t ig = i + lDom[0].first() - nghost;
                const size_t jg = j + lDom[1].first() - nghost;
                const size_t kg = k + lDom[2].first() - nghost;

                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                double u_exact = Kokkos::sin(pi * x) * Kokkos::sin(pi * y) * Kokkos::sin(pi * z);

                viewSol(i, j, k) = u_exact;
                viewRHS(i, j, k) = 3.0 * pi * pi * u_exact;
            });

        // 5. Setup and Run Solver
        ippl::PoissonCG<field_type> lapsolver;

        ippl::ParameterList params;
        params.add("max_iterations", 500);
        params.add("solver", "preconditioned");
        params.add("preconditioner_type", "multigrid");  // Hooks into your new class

        // --- Required by PoissonCG generic setup to prevent IpplException ---
        params.add("newton_level", 5);
        params.add("chebyshev_degree", 31);
        params.add("gauss_seidel_inner_iterations", 2);
        params.add("gauss_seidel_outer_iterations", 2);
        params.add("ssor_omega", 1.57079632679);
        params.add("richardson_iterations", 4);
        params.add("communication", 1);

        // --- MG Parameters ---
        params.add("mg_pre_smooth_iters", 2);
        params.add("mg_post_smooth_iters", 2);
        params.add("mg_omega", 0.8);

        lapsolver.mergeParameters(params);
        lapsolver.setRhs(rhs);
        lapsolver.setLhs(lhs);

        // Initial guess = 0
        lhs = 0.0;

        info << "Starting Conjugate Gradient solve..." << endl;
        IpplTimings::TimerRef solve = IpplTimings::getTimer("solve");
        IpplTimings::startTimer(solve);
        lapsolver.solve();
        IpplTimings::stopTimer(solve);
        info << "Solve completed." << endl;

        // 6. Error Analysis
        field_type error(mesh, layout);

        // Relative error from exact analytical solution
        error           = lhs - solution;
        double relError = norm(error) / norm(solution);

        // Residual error ( Laplacian(u) - RHS )
        // Note: Make sure IPPL laplace operator is imported via IpplOperations.h if needed
        error          = -laplace(lhs) - rhs;
        double residue = norm(error) / norm(rhs);

        int itCount = lapsolver.getIterationCount();

        info << "---------------------------------------" << endl;
        info << "Iterations : " << itCount << endl;
        info << "Rel. Error : " << std::setprecision(8) << relError << endl;
        info << "Residual   : " << std::setprecision(8) << residue << endl;
        info << "---------------------------------------" << endl;

        IpplTimings::print();
    }
    ippl::finalize();

    return 0;
}
