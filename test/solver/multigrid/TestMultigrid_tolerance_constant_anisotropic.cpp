// Continuous manufactured-solution convergence test for the Poisson CG solver
// with homogeneous ConstantFace boundary conditions on all 6 faces.
//
// Purpose:
//   Verify convergence under grid refinement and tightening tolerance for a
//   smooth manufactured solution with anisotropic frequencies.
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
// Manufactured solution (anisotropic frequencies):
//
//       u(x,y,z) = sin(sin(pi x)) sin(sin(2*pi y)) sin(sin(3*pi z)).
//
// Analytical RHS:
//
//       -Delta u = f.
//
// Usage:
//     ./TestMultigrid_convergence_constant_aniso --info 5
//     ./TestMultigrid_convergence_constant_aniso 8 --info 5

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
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
        int maxIterations = 4000;

        if (argc > 1 && argv[1][0] != '-') {
            maxPow = std::atoi(argv[1]);
        }

        Inform m("");
        m << "tolerance,size,h,relError,trueResidual,solverResidual,itCount,solveTime" << endl;

        for (unsigned pt = 1u << 3; pt <= (1u << maxPow); pt = pt << 1) {
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

            auto viewRHS = rhs.getView();
            auto viewSol = solution.getView();

            const auto lDom = layout.getLocalNDIndex();
            const double pi = Kokkos::numbers::pi_v<double>;

            // Different frequencies per dimension
            const double ax_freq = 1.0 * pi;  // x: 1 wavelength
            const double ay_freq = 2.0 * pi;  // y: 2 wavelengths
            const double az_freq = 3.0 * pi;  // z: 3 wavelengths

            using Kokkos::cos;
            using Kokkos::sin;

            const int shift = solution.getNghost();
            auto policy     = solution.getFieldRangePolicy();

            Kokkos::parallel_for(
                "Assign analytical solution and rhs", policy,
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + lDom[0].first() - shift;
                    const int jg = j + lDom[1].first() - shift;
                    const int kg = k + lDom[2].first() - shift;

                    const double x = origin[0] + (static_cast<double>(ig) + 1.0) * hx[0];
                    const double y = origin[1] + (static_cast<double>(jg) + 1.0) * hx[1];
                    const double z = origin[2] + (static_cast<double>(kg) + 1.0) * hx[2];

                    const double ax = ax_freq * x;
                    const double ay = ay_freq * y;
                    const double az = az_freq * z;

                    const double sx = sin(sin(ax));
                    const double sy = sin(sin(ay));
                    const double sz = sin(sin(az));

                    const double cx = cos(sin(ax));
                    const double cy = cos(sin(ay));
                    const double cz = cos(sin(az));

                    const double sax = sin(ax);
                    const double say = sin(ay);
                    const double saz = sin(az);

                    const double cax = cos(ax);
                    const double cay = cos(ay);
                    const double caz = cos(az);

                    const double u = sx * sy * sz;

                    const double f =
                        ax_freq * ax_freq * cx * sax * sy * sz
                        + ay_freq * ay_freq * sx * cy * say * sz
                        + az_freq * az_freq * sx * sy * cz * saz
                        + (ax_freq * ax_freq * cax * cax + ay_freq * ay_freq * cay * cay
                           + az_freq * az_freq * caz * caz)
                              * sx * sy * sz;

                    viewSol(i, j, k) = u;
                    viewRHS(i, j, k) = f;
                });

            solution.fillHalo();
            rhs.fillHalo();

            for (double tolerance = 1e-5; tolerance >= 1e-13; tolerance *= 1e-2) {
                lhs = 0.0;

                ippl::PoissonCG<Field_t> solver;
                ippl::ParameterList params;

                params.add("max_iterations", maxIterations);
                params.add("tolerance", tolerance);

                params.add("solver", "preconditioned");
                params.add("preconditioner_type", "multigrid");

                params.add("newton_level", 5);
                params.add("chebyshev_degree", 31);
                params.add("gauss_seidel_inner_iterations", 2);
                params.add("gauss_seidel_outer_iterations", 2);
                params.add("ssor_omega", 1.57079632679);
                params.add("richardson_iterations", 4);
                params.add("communication", 0);

                params.add("mg_pre_smooth_iters", 2);
                params.add("mg_post_smooth_iters", 2);
                params.add("mg_omega", 0.8);
                params.add("min_cells_per_rank_per_dim", 2);

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

                m << std::scientific << std::setprecision(1) << tolerance << ","
                  << std::defaultfloat << pt << "," << std::setprecision(16) << dx << ","
                  << relError << "," << trueResidual << "," << solverResidual << "," << itCount
                  << "," << solveTime << endl;
            }
        }
    }
    ippl::finalize();

    return 0;
}
