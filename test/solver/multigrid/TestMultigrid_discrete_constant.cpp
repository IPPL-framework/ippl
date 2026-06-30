// Discrete manufactured-solution test for the Poisson CG solver with
// homogeneous ConstantFace boundary conditions on all 6 faces.
//
// Purpose:
//   Verify algebraic correctness of the discrete operator/solver combination.
//   We prescribe an exact grid function u_ex, then build the RHS with the same
//   discrete operator used by the solver:
//
//       f_h = -Delta_h u_ex .
//
// Thus, up to solver tolerance, the numerical solution should reproduce u_ex.
//
// Ghost-value convention:
//   IPPL's current ConstantFace(0.0) sets the ghost cell value directly.
//   Therefore we use the shifted coordinates
//
//       h   = 1 / (N + 1),
//       x_i = (i + 1) h,
//
// so that the lower/upper ghost layers correspond to x = 0 and x = 1.
//
// Exact discrete solution:
//
//       u_ex(x,y,z) = x (1-x) y (1-y) z (1-z).
//
// Usage:
//     ./TestMultigrid_discrete_constant --info 5
//     ./TestMultigrid_discrete_constant 8 --info 5
//
// Optional first positional argument:
//     maxPow
//
// Default maxPow = 6, i.e. sizes 4^3, 8^3, ..., 128^3.

#include "Ippl.h"

#include <array>
#include <cstdlib>
#include <iomanip>

#include "Utility/Inform.h"

#include "PoissonSolvers/PoissonCG.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;

        using Mesh_t      = ippl::UniformCartesian<double, dim>;
        using Centering_t = Mesh_t::DefaultCentering;
        using Field_t     = ippl::Field<double, dim, Mesh_t, Centering_t>;
        using BConds_t    = ippl::BConds<Field_t, dim>;

        int maxPow        = 7;
        double tolerance  = 1e-13;
        int maxIterations = 2000;

        if (argc > 1 && argv[1][0] != '-') {
            maxPow = std::atoi(argv[1]);
        }

        Inform m("");
        m << "solver,size,h,relError,trueResidual,solverResidual,itCount,solveTime" << endl;

        for (unsigned pt = 1u << 2; pt <= (1u << maxPow); pt = pt << 1) {
            ippl::Vector<unsigned, dim> I(pt);
            ippl::NDIndex<dim> domain(I);

            std::array<bool, dim> isParallel;
            isParallel.fill(true);

            ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, domain, isParallel);

            const double dx = 1.0 / static_cast<double>(pt + 1);

            ippl::Vector<double, dim> hx     = dx;
            ippl::Vector<double, dim> origin = 0.0;

            Mesh_t mesh(domain, hx, origin);

            Field_t rhs(mesh, layout);
            Field_t lhs(mesh, layout);
            Field_t solution(mesh, layout);

            BConds_t bcField;
            for (unsigned int i = 0; i < 2 * dim; ++i) {
                bcField[i] = std::make_shared<ippl::ConstantFace<Field_t>>(i, 0.0);
            }

            lhs.setFieldBC(bcField);
            rhs.setFieldBC(bcField);
            solution.setFieldBC(bcField);

            auto viewSol = solution.getView();

            const auto lDom = layout.getLocalNDIndex();
            const int shift = solution.getNghost();
            auto policy     = solution.getFieldRangePolicy();

            Kokkos::parallel_for(
                "Assign exact discrete solution", policy,
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + lDom[0].first() - shift;
                    const int jg = j + lDom[1].first() - shift;
                    const int kg = k + lDom[2].first() - shift;

                    const double x = origin[0] + (static_cast<double>(ig) + 1.0) * hx[0];
                    const double y = origin[1] + (static_cast<double>(jg) + 1.0) * hx[1];
                    const double z = origin[2] + (static_cast<double>(kg) + 1.0) * hx[2];

                    const double u   = x * (1.0 - x) * y * (1.0 - y) * z * (1.0 - z);
                    viewSol(i, j, k) = u;
                });

            solution.fillHalo();

            rhs = -laplace(solution);
            rhs.fillHalo();

            for (int mode = 0; mode < 2; ++mode) {
                const bool useMG      = mode == 1;
                const char* solverTag = useMG ? "mg_pcg" : "cg";

                lhs = 0.0;

                ippl::PoissonCG<Field_t> solver;
                ippl::ParameterList params;

                params.add("max_iterations", maxIterations);
                params.add("tolerance", tolerance);

                if (useMG) {
                    params.add("solver", "preconditioned");
                    params.add("preconditioner_type", "multigrid");

                    params.add("newton_level", 5);
                    params.add("chebyshev_degree", 31);
                    params.add("gauss_seidel_inner_iterations", 2);
                    params.add("gauss_seidel_outer_iterations", 2);
                    params.add("ssor_omega", 1.57079632679);
                    params.add("richardson_iterations", 4);
                    params.add("communication", 1);

                    params.add("mg_pre_smooth_iters", 2);
                    params.add("mg_post_smooth_iters", 2);
                    params.add("mg_omega", 0.8);
                    params.add("min_cells_per_rank_per_dim", 2);
                } else {
                    params.add("solver", "non-preconditioned");
                }

                solver.mergeParameters(params);
                solver.setRhs(rhs);
                solver.setLhs(lhs);

                ippl::Comm->barrier();
                const double t0 = MPI_Wtime();

                solver.solve();

                ippl::Comm->barrier();
                const double t1 = MPI_Wtime();

                lhs.fillHalo();

                Field_t error(mesh, layout);

                error                 = lhs - solution;
                const double relError = norm(error) / norm(solution);

                error                     = -laplace(lhs) - rhs;
                const double trueResidual = norm(error) / norm(rhs);

                const double solverResidual = solver.getResidue();
                const int itCount           = solver.getIterationCount();
                const double solveTime      = t1 - t0;

                m << solverTag << "," << pt << "," << std::setprecision(16) << dx << "," << relError
                  << "," << trueResidual << "," << solverResidual << "," << itCount << ","
                  << solveTime << endl;
            }
        }
    }
    ippl::finalize();

    return 0;
}
