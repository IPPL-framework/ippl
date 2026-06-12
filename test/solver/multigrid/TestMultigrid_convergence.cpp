// Tests the Poisson CG solver for a manufactured solution with
// homogeneous ConstantFace boundary conditions on all 6 faces.
//
// For each grid size, this test runs:
//   1) plain CG  (non-preconditioned)
//   2) MG-PCG    (multigrid preconditioned CG)
//
// Manufactured solution on [0,1]^3:
//     u(x,y,z) = sin(pi x) sin(pi y) sin(pi z)
// so that
//     -Delta u = 3 pi^2 u
//
// Usage:
//     ./TestMultigrid_convergence.cpp --info 5
//
// Note:
// * The default max size is kept moderate because plain 3D CG becomes
//   expensive quickly without preconditioning.

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <array>
#include <cstdlib>
#include <iomanip>

#include "Utility/Inform.h"
#include "Utility/IpplTimings.h"

#include "PoissonSolvers/PoissonCG.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, dim>;
        using Centering_t          = Mesh_t::DefaultCentering;
        using field_type           = ippl::Field<double, dim, Mesh_t, Centering_t>;
        using bc_type              = ippl::BConds<field_type, dim>;

        int maxPow        = 8;
        double tolerance  = 1e-13;
        int maxIterations = 5000;

        Inform m("");
        m << "solver,size,relError,trueResidual,solverResidual,itCount,solveTime" << endl;

        for (unsigned pt = 1 << 2; pt <= (1u << maxPow); pt = pt << 1) {
            ippl::Vector<unsigned, dim> I(pt);
            ippl::NDIndex<dim> domain(I);

            std::array<bool, dim> isParallel;
            for (unsigned int d = 0; d < dim; ++d) {
                isParallel[d] = true;
            }

            ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, domain, isParallel);

            // Unit cube [0,1]^3
            double dx                        = 1.0 / double(pt);
            ippl::Vector<double, dim> hx     = dx;
            ippl::Vector<double, dim> origin = 0.0;
            Mesh_t mesh(domain, hx, origin);

            field_type rhs(mesh, layout), lhs(mesh, layout), solution(mesh, layout);

            bc_type bcField;
            for (unsigned int i = 0; i < 2 * dim; ++i) {
                bcField[i] = std::make_shared<ippl::ConstantFace<field_type>>(i, 0.0);
            }

            lhs.setFieldBC(bcField);
            rhs.setFieldBC(bcField);
            solution.setFieldBC(bcField);

            auto& viewRHS = rhs.getView();
            auto& viewSol = solution.getView();

            const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();
            const double pi                = Kokkos::numbers::pi_v<double>;

            using Kokkos::sin;

            const int shiftSol = solution.getNghost();
            auto policySol     = solution.getFieldRangePolicy();
            Kokkos::parallel_for(
                "Assign solution", policySol, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + lDom[0].first() - shiftSol;
                    const int jg = j + lDom[1].first() - shiftSol;
                    const int kg = k + lDom[2].first() - shiftSol;

                    const double x = origin[0] + (ig + 0.5) * hx[0];
                    const double y = origin[1] + (jg + 0.5) * hx[1];
                    const double z = origin[2] + (kg + 0.5) * hx[2];

                    viewSol(i, j, k) = sin(pi * x) * sin(pi * y) * sin(pi * z);
                });

            const int shiftRHS = rhs.getNghost();
            auto policyRHS     = rhs.getFieldRangePolicy();
            Kokkos::parallel_for(
                "Assign rhs", policyRHS, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + lDom[0].first() - shiftRHS;
                    const int jg = j + lDom[1].first() - shiftRHS;
                    const int kg = k + lDom[2].first() - shiftRHS;

                    const double x = origin[0] + (ig + 0.5) * hx[0];
                    const double y = origin[1] + (jg + 0.5) * hx[1];
                    const double z = origin[2] + (kg + 0.5) * hx[2];

                    const double u   = sin(pi * x) * sin(pi * y) * sin(pi * z);
                    viewRHS(i, j, k) = 3.0 * pi * pi * u;
                });

            for (int mode = 0; mode < 2; ++mode) {
                const bool useMG      = (mode == 1);
                const char* solverTag = useMG ? "mg_pcg" : "cg";

                lhs = 0.0;

                ippl::PoissonCG<field_type> lapsolver;
                ippl::ParameterList params;

                params.add("max_iterations", maxIterations);
                params.add("tolerance", tolerance);

                if (useMG) {
                    params.add("solver", "preconditioned");
                    params.add("preconditioner_type", "multigrid");

                    // Required by PoissonCG generic setup
                    params.add("newton_level", 5);
                    params.add("chebyshev_degree", 31);
                    params.add("gauss_seidel_inner_iterations", 2);
                    params.add("gauss_seidel_outer_iterations", 2);
                    params.add("ssor_omega", 1.57079632679);
                    params.add("richardson_iterations", 4);
                    params.add("communication", 1);

                    // MG parameters
                    params.add("mg_pre_smooth_iters", 2);
                    params.add("mg_post_smooth_iters", 2);
                    params.add("mg_omega", 0.8);
                    params.add("min_cells_per_rank_per_dim", 2);
                } else {
                    params.add("solver", "non-preconditioned");
                }

                lapsolver.mergeParameters(params);
                lapsolver.setRhs(rhs);
                lapsolver.setLhs(lhs);

                ippl::Comm->barrier();
                double t0 = MPI_Wtime();

                lapsolver.solve();

                ippl::Comm->barrier();
                double t1        = MPI_Wtime();
                double solveTime = t1 - t0;

                field_type error(mesh, layout);

                error           = lhs - solution;
                double relError = norm(error) / norm(solution);

                error               = -laplace(lhs) - rhs;
                double trueResidual = norm(error) / norm(rhs);

                // Note:
                // * for plain CG this is the solver's internal residual norm
                // * for PCG this is the preconditioned residual quantity
                double solverResidual = lapsolver.getResidue();

                int itCount = lapsolver.getIterationCount();

                m << solverTag << "," << pt << "," << std::setprecision(16) << relError << ","
                  << trueResidual << "," << solverResidual << "," << itCount << "," << solveTime
                  << endl;
            }
        }

        IpplTimings::print("timings_constant_cg_mg.dat");
    }
    ippl::finalize();

    return 0;
}
