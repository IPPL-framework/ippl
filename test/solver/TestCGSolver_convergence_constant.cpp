// Continuous manufactured-solution convergence test for the Poisson CG solver
// with homogeneous ConstantFace boundary conditions on all 6 faces.
//
// Purpose:
//   Verify convergence under grid refinement for a smooth manufactured
//   solution that is not a Laplacian eigenfunction.
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
// Manufactured solution:
//
//       u(x,y,z) = sin(sin(pi x)) sin(sin(pi y)) sin(sin(pi z)).
//
// Analytical RHS:
//
//       -Delta u = f.
//
// Usage:
//     ./TestCGSolver_convergence_constant --info 5
//     ./TestCGSolver_convergence_constant 8 --info 5
//
// Optional first positional argument:
//     maxPow
//
// Default maxPow = 6, i.e. sizes 4^3, 8^3, ..., 128^3.

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
        double tolerance  = 1e-13;
        int maxIterations = 4000;

        if (argc > 1 && argv[1][0] != '-') {
            maxPow = std::atoi(argv[1]);
        }

        Inform m("");
        m << "size, relError, residue, itCount, h" << endl;

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

            auto viewRHS = rhs.getView();
            auto viewSol = solution.getView();

            const auto lDom = layout.getLocalNDIndex();
            const double pi = Kokkos::numbers::pi_v<double>;

            using Kokkos::cos;
            using Kokkos::sin;

            int shift1     = solution.getNghost();
            auto policySol = solution.getFieldRangePolicy();

            Kokkos::parallel_for(
                "Assign solution", policySol, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + lDom[0].first() - shift1;
                    const int jg = j + lDom[1].first() - shift1;
                    const int kg = k + lDom[2].first() - shift1;

                    const double x = origin[0] + (static_cast<double>(ig) + 1.0) * hx[0];
                    const double y = origin[1] + (static_cast<double>(jg) + 1.0) * hx[1];
                    const double z = origin[2] + (static_cast<double>(kg) + 1.0) * hx[2];

                    viewSol(i, j, k) = sin(sin(pi * x)) * sin(sin(pi * y)) * sin(sin(pi * z));
                });

            const int shift2 = rhs.getNghost();
            auto policyRHS   = rhs.getFieldRangePolicy();

            Kokkos::parallel_for(
                "Assign rhs", policyRHS, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + lDom[0].first() - shift2;
                    const int jg = j + lDom[1].first() - shift2;
                    const int kg = k + lDom[2].first() - shift2;

                    const double x = origin[0] + (static_cast<double>(ig) + 1.0) * hx[0];
                    const double y = origin[1] + (static_cast<double>(jg) + 1.0) * hx[1];
                    const double z = origin[2] + (static_cast<double>(kg) + 1.0) * hx[2];

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

            solution.fillHalo();
            rhs.fillHalo();

            lhs = 0.0;

            ippl::PoissonCG<Field_t> solver;
            ippl::ParameterList params;

            params.add("max_iterations", maxIterations);
            params.add("tolerance", tolerance);
            params.add("solver", "non-preconditioned");

            solver.mergeParameters(params);
            solver.setRhs(rhs);
            solver.setLhs(lhs);

            solver.solve();

            lhs.fillHalo();

            Field_t error(mesh, layout);

            error                 = lhs - solution;
            const double relError = norm(error) / norm(solution);

            error                = -laplace(lhs) - rhs;
            const double residue = norm(error) / norm(rhs);

            const int itCount = solver.getIterationCount();

            m << pt << ", " << std::setprecision(16) << relError << ", " << residue << ", "
              << itCount << ", " << dx << endl;
        }
    }
    ippl::finalize();

    return 0;
}
