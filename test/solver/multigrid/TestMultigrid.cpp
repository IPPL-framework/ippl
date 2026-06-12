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
#include <iostream>
#include <string>
#include <vector>

#include "Utility/Inform.h"
#include "Utility/IpplTimings.h"

#include "PoissonSolvers/PoissonCG.h"

template <typename FieldType>
void printGlobalField(const FieldType& field, const std::string& name) {
    // 1. Get global dimensions
    auto& layout = field.getLayout();
    auto& domain = layout.getDomain();

    int Nx = domain[0].length();
    int Ny = domain[1].length();
    int Nz = domain[2].length();

    // 2. Allocate a global-sized buffer initialized to 0.0 on all ranks
    std::vector<double> local_buffer(Nx * Ny * Nz, 0.0);

    // 3. Get local bounds and ghost information
    auto& lDom = layout.getLocalNDIndex();
    int nghost = field.getNghost();

    // 4. Safely pull data from GPU/Device to Host
    auto view  = field.getView();
    auto hview = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(hview, view);

    // 5. Fill the buffer ONLY with the physical cells owned by this rank
    for (size_t i = 0; i < lDom[0].length(); ++i) {
        for (size_t j = 0; j < lDom[1].length(); ++j) {
            for (size_t k = 0; k < lDom[2].length(); ++k) {
                // Calculate global indices
                int ig = lDom[0].first() + i;
                int jg = lDom[1].first() + j;
                int kg = lDom[2].first() + k;

                // Flatten to 1D index
                int flat_idx = ig * (Ny * Nz) + jg * Nz + kg;

                // Read from host view (offset by nghost to skip the boundary halo)
                local_buffer[flat_idx] = hview(i + nghost, j + nghost, k + nghost);
            }
        }
    }

    // 6. Reduce (sum) all local buffers into a single master buffer on Rank 0
    std::vector<double> global_buffer;
    if (ippl::Comm->rank() == 0) {
        global_buffer.resize(Nx * Ny * Nz, 0.0);
    }

    MPI_Reduce(local_buffer.data(), ippl::Comm->rank() == 0 ? global_buffer.data() : nullptr,
               Nx * Ny * Nz, MPI_DOUBLE, MPI_SUM, 0, ippl::Comm->getCommunicator());

    // 7. Rank 0 prints the assembled field in the same loop order IPPL natively uses
    if (ippl::Comm->rank() == 0) {
        std::cout << name << ":\n";
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    int flat_idx = i * (Ny * Nz) + j * Nz + k;
                    std::cout << global_buffer[flat_idx] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    // Synchronize ranks before continuing
    ippl::fence();
    ippl::Comm->barrier();
}

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
        int Nx        = 32;
        int Ny        = 32;
        int Nz        = 32;
        int test_case = 1;  // 1 = Periodic, 2 = Mixed BCs, 3 or more = constant

        if (argc >= 4) {
            Nx = std::atoi(argv[1]);
            Ny = std::atoi(argv[2]);
            Nz = std::atoi(argv[3]);
        } else if (argc >= 2) {
            // Fallback: single argument interpreted as log2 size (legacy)
            int N = static_cast<int>(std::strtol(argv[1], NULL, 10));
            Nx = Ny = Nz = 1 << N;
            info << "Warning: single argument interpreted as log2 size -> " << Nx << "^3" << endl;
        }
        if (argc >= 5) {
            test_case = std::atoi(argv[4]);
        }
        const char* tc_name = (test_case == 1)   ? " (All Periodic)"
                              : (test_case == 2) ? " (Mixed Periodic/Dirichlet)"
                                                 : " (All ConstantFace)";

        info << "Grid Size: " << Nx << " x " << Ny << " x " << Nz << endl;
        info << "Test Case: " << test_case << tc_name << endl;

        // 1. Setup Domain and Layout
        ippl::Index Ix(Nx), Iy(Ny), Iz(Nz);
        ippl::NDIndex<dim> owned(Ix, Iy, Iz);
        std::array<bool, dim> isParallel = {true, true, true};
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        // 2. Setup Mesh, Spacing, and Boundary Conditions based on Test Case
        double dx, dy, dz;
        ippl::Vector<double, dim> origin = 0.0;
        bc_type bcField;

        if (test_case == 1) {
            dx = 2.0 / double(Nx);
            dy = 2.0 / double(Ny);
            dz = 2.0 / double(Nz);
            for (unsigned int i = 0; i < 2 * dim; ++i) {
                bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
            }
        } else if (test_case == 2) {
            dx = 2.0 / double(Nx);
            dy = 1.0 / double(Ny);
            dz = 1.0 / double(Nz);

            bcField[0] = std::make_shared<ippl::PeriodicFace<field_type>>(0);
            bcField[1] = std::make_shared<ippl::PeriodicFace<field_type>>(1);
            bcField[2] = std::make_shared<ippl::ZeroFace<field_type>>(2);
            bcField[3] = std::make_shared<ippl::ZeroFace<field_type>>(3);
            bcField[4] = std::make_shared<ippl::ZeroFace<field_type>>(4);
            bcField[5] = std::make_shared<ippl::ZeroFace<field_type>>(5);
        } else {
            dx = 1.0 / double(Nx);
            dy = 1.0 / double(Ny);
            dz = 1.0 / double(Nz);
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
        params.add("preconditioner_type", "multigrid");

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
        params.add("min_cells_per_rank_per_dim", 2);

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

        // printGlobalField(solution, "Solution");
        // printGlobalField(lhs, "Approx");

        IpplTimings::print();
    }
    ippl::finalize();

    return 0;
}
