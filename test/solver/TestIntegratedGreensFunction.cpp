/**
 * @file TestIntegratedGreensFunction.cpp
 * @brief Regression test for integrated and shifted integrated Hockney kernels.
 *
 * This test exercises FFTOpenPoissonSolver with
 * `greens_function = Solver_t::INTEGRATED`, `algorithm = Solver_t::HOCKNEY`,
 * and `output_type = Solver_t::SOL`.
 *
 * The source is deliberately minimal: one physical mesh cell at the origin is
 * assigned unit density and every other cell is zero. With IPPL's Hockney solve
 * normalization, the potential sampled at physical grid offset
 * @f$\mathbf{r}_{ijk}=(i h_x, j h_y, k h_z)@f$ is expected to be
 * @f[
 *   \phi_{ijk} =
 *   \frac{h_x h_y h_z}{4\pi}
 *   \frac{1}{h_x h_y h_z}
 *   \int_{\mathrm{cell}}
 *   \frac{d^3\mathbf{r}'}{|\mathbf{r}_{ijk}-\mathbf{r}'|}.
 * @f]
 *
 * The shifted case calls FFTOpenPoissonSolver::shiftedGreensFunction() and
 * checks the translated cell average
 * @f[
 *   \phi^s_{ijk} =
 *   \frac{h_x h_y h_z}{4\pi}
 *   \frac{1}{h_x h_y h_z}
 *   \int_{\mathrm{cell}}
 *   \frac{d^3\mathbf{r}'}{|\mathbf{r}_{ijk}-\mathbf{s}-\mathbf{r}'|}.
 * @f]
 *
 * The reference values are computed independently in this file from the same
 * closed-form eight-corner antiderivative used by Qiang et al. This is a
 * deterministic kernel/normalization test rather than a beam-physics
 * convergence benchmark; it is intended to catch sign, shift, cell-volume, and
 * distributed-layout mistakes.
 */

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cmath>
#include <functional>

#include "PoissonSolvers/FFTOpenPoissonSolver.h"

namespace {
    // Principal arctangent used by the closed-form antiderivative. The
    // denominator can vanish on coordinate planes; returning the limiting
    // +/-pi/2 value avoids NaNs while preserving atan(numerator/denominator)
    // semantics.
    KOKKOS_INLINE_FUNCTION double atan_ratio(double numerator, double denominator) {
        const double pi = Kokkos::numbers::pi_v<double>;
        if (denominator == 0.0) {
            if (numerator > 0.0)
                return 0.5 * pi;
            if (numerator < 0.0)
                return -0.5 * pi;
            return 0.0;
        }
        return Kokkos::atan(numerator / denominator);
    }

    // Term b*c*log(a+r). The coefficient is exactly zero on coordinate planes,
    // where a+r can also be zero, so skip the logarithm in that removable case.
    KOKKOS_INLINE_FUNCTION double log_term(double a, double b, double c, double r) {
        const double coeff = b * c;
        return (coeff == 0.0) ? 0.0 : coeff * Kokkos::log(a + r);
    }

    // Antiderivative F(x,y,z) for the volume integral of 1/r over a rectangular
    // cell. The cell integral is the signed eight-corner difference of F.
    KOKKOS_INLINE_FUNCTION double antiderivative(double x, double y, double z) {
        const double r2 = x * x + y * y + z * z;
        if (r2 == 0.0)
            return 0.0;

        const double r = Kokkos::sqrt(r2);
        double value   = 0.0;
        value -= 0.5 * z * z * atan_ratio(x * y, z * r);
        value -= 0.5 * y * y * atan_ratio(x * z, y * r);
        value -= 0.5 * x * x * atan_ratio(y * z, x * r);
        value += log_term(x, y, z, r);
        value += log_term(y, x, z, r);
        value += log_term(z, x, y, r);
        return value;
    }

    // Average of 1/r over a cell of size hx*hy*hz centered at (x,y,z). This
    // helper deliberately returns the average without the 1/(4*pi) prefactor;
    // the test applies that prefactor at the same point where the solver does.
    KOKKOS_INLINE_FUNCTION double integrated_average(double x, double y, double z, double hx,
                                                     double hy, double hz) {
        const double x0 = x - 0.5 * hx;
        const double x1 = x + 0.5 * hx;
        const double y0 = y - 0.5 * hy;
        const double y1 = y + 0.5 * hy;
        const double z0 = z - 0.5 * hz;
        const double z1 = z + 0.5 * hz;

        double integral = antiderivative(x1, y1, z1);
        integral -= antiderivative(x0, y1, z1);
        integral -= antiderivative(x1, y0, z1);
        integral += antiderivative(x0, y0, z1);
        integral -= antiderivative(x1, y1, z0);
        integral += antiderivative(x0, y1, z0);
        integral += antiderivative(x1, y0, z0);
        integral -= antiderivative(x0, y0, z0);

        return integral / (hx * hy * hz);
    }
}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int exit_code = 0;
    {
        Inform msg("TestIntegratedGreensFunction");

        constexpr unsigned int Dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, Dim>;
        using Centering_t          = Mesh_t::DefaultCentering;
        using field_t              = ippl::Field<double, Dim, Mesh_t, Centering_t>;
        using fieldV_t = ippl::Field<ippl::Vector<double, Dim>, Dim, Mesh_t, Centering_t>;
        using Solver_t = ippl::FFTOpenPoissonSolver<fieldV_t, field_t>;

        const int N = 8;
        ippl::NDIndex<Dim> owned;
        for (unsigned d = 0; d < Dim; ++d) {
            owned[d] = ippl::Index(N);
        }

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        ippl::Vector<double, Dim> hr     = {0.17, 0.23, 0.31};
        ippl::Vector<double, Dim> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hr, origin);
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel);

        field_t rho, rho_clean, phi_unshifted, phi_shifted;
        rho.initialize(mesh, layout);
        rho_clean.initialize(mesh, layout);
        phi_unshifted.initialize(mesh, layout);
        phi_shifted.initialize(mesh, layout);

        // Put unit density in the global origin cell. The physical source
        // charge is therefore rho * cellVolume = cellVolume, which is why the
        // expected potential below carries an explicit cellVolume factor.
        auto initialize_delta = [&]() {
            auto view        = rho.getView();
            auto clean       = rho_clean.getView();
            const int nghost = rho.getNghost();
            const auto& ldom = layout.getLocalNDIndex();
            Kokkos::parallel_for(
                "initialize unit cell density", rho.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig       = i + ldom[0].first() - nghost;
                    const int jg       = j + ldom[1].first() - nghost;
                    const int kg       = k + ldom[2].first() - nghost;
                    const double value = (ig == 0 && jg == 0 && kg == 0) ? 1.0 : 0.0;
                    view(i, j, k)      = value;
                    clean(i, j, k)     = value;
                });
        };

        ippl::ParameterList params;
        params.add("use_pencils", true);
        params.add("comm", ippl::a2a);
        params.add("use_reorder", false);
        params.add("use_heffte_defaults", false);
        params.add("use_gpu_aware", true);
        params.add("r2c_direction", 0);
        params.add("algorithm", Solver_t::HOCKNEY);
        params.add("greens_function", Solver_t::INTEGRATED);
        params.add("output_type", Solver_t::SOL);

        initialize_delta();
        Solver_t solver(rho, params);
        solver.solve();
        Kokkos::deep_copy(phi_unshifted.getView(), rho.getView());

        Kokkos::deep_copy(rho.getView(), rho_clean.getView());
        const ippl::Vector<double, Dim> shift = {0.41, -0.19, 0.37};
        solver.shiftedGreensFunction(shift);
        solver.solve();
        Kokkos::deep_copy(phi_shifted.getView(), rho.getView());

        const double cellVolume = hr[0] * hr[1] * hr[2];
        const double inv4pi     = 1.0 / (4.0 * Kokkos::numbers::pi_v<double>);

        // Compare a solved potential against the analytical integrated kernel
        // on the physical N^3 domain. kernelShift = 0 tests greensFunction();
        // kernelShift = shift tests shiftedGreensFunction(shift).
        auto compute_error = [&](field_t& phi, const ippl::Vector<double, Dim>& kernelShift) {
            auto view        = phi.getView();
            const int nghost = phi.getNghost();
            const auto& ldom = layout.getLocalNDIndex();
            double localErr  = 0.0;
            double localRef  = 0.0;

            Kokkos::parallel_reduce(
                "integrated green max error", phi.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& lmax) {
                    const int ig   = i + ldom[0].first() - nghost;
                    const int jg   = j + ldom[1].first() - nghost;
                    const int kg   = k + ldom[2].first() - nghost;
                    const double x = static_cast<double>(ig) * hr[0] - kernelShift[0];
                    const double y = static_cast<double>(jg) * hr[1] - kernelShift[1];
                    const double z = static_cast<double>(kg) * hr[2] - kernelShift[2];
                    // solve() multiplies the inverse FFT result by the cell
                    // volume. The integrated kernel itself is a cell average.
                    const double expected =
                        cellVolume * inv4pi * integrated_average(x, y, z, hr[0], hr[1], hr[2]);
                    const double diff = Kokkos::fabs(view(i, j, k) - expected);
                    if (diff > lmax)
                        lmax = diff;
                },
                Kokkos::Max<double>(localErr));

            Kokkos::parallel_reduce(
                "integrated green max reference", phi.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& lmax) {
                    const int ig   = i + ldom[0].first() - nghost;
                    const int jg   = j + ldom[1].first() - nghost;
                    const int kg   = k + ldom[2].first() - nghost;
                    const double x = static_cast<double>(ig) * hr[0] - kernelShift[0];
                    const double y = static_cast<double>(jg) * hr[1] - kernelShift[1];
                    const double z = static_cast<double>(kg) * hr[2] - kernelShift[2];
                    const double expected =
                        cellVolume * inv4pi * integrated_average(x, y, z, hr[0], hr[1], hr[2]);
                    const double absv = Kokkos::fabs(expected);
                    if (absv > lmax)
                        lmax = absv;
                },
                Kokkos::Max<double>(localRef));

            double globalErr = 0.0;
            double globalRef = 0.0;
            ippl::Comm->allreduce(localErr, globalErr, 1, std::greater<double>());
            ippl::Comm->allreduce(localRef, globalRef, 1, std::greater<double>());
            return std::pair<double, double>(globalErr, globalRef);
        };

        const ippl::Vector<double, Dim> zeroShift = {0.0, 0.0, 0.0};
        const auto [unshiftedErr, unshiftedRef]   = compute_error(phi_unshifted, zeroShift);
        const auto [shiftedErr, shiftedRef]       = compute_error(phi_shifted, shift);
        const double unshiftedRel                 = unshiftedErr / unshiftedRef;
        const double shiftedRel                   = shiftedErr / shiftedRef;

        msg << "unshifted max abs error = " << unshiftedErr << ", relative = " << unshiftedRel
            << endl;
        msg << "shifted max abs error = " << shiftedErr << ", relative = " << shiftedRel << endl;

        // This is an exact closed-form kernel check up to FFT and floating-point
        // roundoff, so the tolerance can be much tighter than a physics
        // convergence test.
        const double tol = 5.0e-12;
        if (unshiftedRel > tol) {
            msg << "FAIL: unshifted integrated Green's function error exceeds " << tol << endl;
            exit_code = 1;
        } else if (shiftedRel > tol) {
            msg << "FAIL: shifted integrated Green's function error exceeds " << tol << endl;
            exit_code = 1;
        } else {
            msg << "PASS" << endl;
        }
    }
    ippl::finalize();
    return exit_code;
}
