/**
 * @file TestIntegratedGreensFunction.cpp
 * @brief Regression test for integrated and shifted integrated Hockney kernels.
 *
 * This test exercises FFTOpenPoissonSolver with
 * `greens_function = Solver_t::INTEGRATED`, `algorithm = Solver_t::HOCKNEY`,
 * and `output_type = Solver_t::SOL`.
 *
 * The source is deliberately minimal: one physical mesh cell @f$\mathbf{s}@f$
 * is assigned unit density and every other cell is zero. With IPPL's Hockney
 * solve normalization, the potential sampled at the grid offset
 * @f$\mathbf{r}_{ijk}=((i-s_x) h_x, (j-s_y) h_y, (k-s_z) h_z)@f$ is expected to be
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
 *   \frac{d^3\mathbf{r}'}{|\mathbf{r}_{ijk}-\mathbf{t}-\mathbf{r}'|}.
 * @f]
 *
 * Two meshes are checked. A near-cubic mesh with the source at the origin
 * catches sign, shift, cell-volume, and distributed-layout mistakes. An
 * elongated mesh (@f$h_z/h_x = 10^5@f$) with the source in the far corner
 * makes every kernel offset negative; there the closed form loses accuracy
 * if @f$\ln(a+r)@f$ is evaluated for @f$a<0@f$, @f$|a|\gg r-|a|@f$, where
 * @f$a+r@f$ cancels (IPPL issue 627).
 *
 * The reference is computed in this file from the closed-form antiderivative
 * of Qiang et al., evaluated only at non-negative arguments: the cell centre
 * is reflected to @f$(|x|,|y|,|z|)@f$, and a cell that straddles a coordinate
 * plane is split there, its negative part mirrored. Both steps are exact
 * because @f$1/r@f$ is even in each coordinate. This is a deterministic
 * kernel/normalization test rather than a beam-physics convergence benchmark.
 */

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cmath>
#include <functional>

#include "PoissonSolvers/FFTOpenPoissonSolver.h"

namespace {
    constexpr unsigned int Dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, Dim>;
    using Centering_t          = Mesh_t::DefaultCentering;
    using field_t              = ippl::Field<double, Dim, Mesh_t, Centering_t>;
    using fieldV_t             = ippl::Field<ippl::Vector<double, Dim>, Dim, Mesh_t, Centering_t>;
    using Solver_t             = ippl::FFTOpenPoissonSolver<fieldV_t, field_t>;
    using Vector_t             = ippl::Vector<double, Dim>;
    using IndexVector          = ippl::Vector<int, Dim>;

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
    // Only called with a >= 0, where a+r does not cancel.
    KOKKOS_INLINE_FUNCTION double log_term(double a, double b, double c, double r) {
        const double coeff = b * c;
        return (coeff == 0.0) ? 0.0 : coeff * Kokkos::log(a + r);
    }

    // Antiderivative F(x,y,z) for the volume integral of 1/r over a rectangular
    // box. The box integral is the signed eight-corner difference of F.
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

    // Integral of 1/r over [x0,x1] x [y0,y1] x [z0,z1] with all bounds >= 0.
    KOKKOS_INLINE_FUNCTION double box_integral(double x0, double x1, double y0, double y1,
                                               double z0, double z1) {
        double integral = antiderivative(x1, y1, z1);
        integral -= antiderivative(x0, y1, z1);
        integral -= antiderivative(x1, y0, z1);
        integral += antiderivative(x0, y0, z1);
        integral -= antiderivative(x1, y1, z0);
        integral += antiderivative(x0, y1, z0);
        integral += antiderivative(x1, y0, z0);
        integral -= antiderivative(x0, y0, z0);
        return integral;
    }

    // Covers the interval [c - h/2, c + h/2], c >= 0, by intervals in [0, inf).
    // If the interval straddles zero, its negative part [c - h/2, 0] is replaced
    // by the mirrored interval [0, h/2 - c].
    KOKKOS_INLINE_FUNCTION int nonnegative_segments(double c, double h, double lo[2],
                                                    double hi[2]) {
        if (c >= 0.5 * h) {
            lo[0] = c - 0.5 * h;
            hi[0] = c + 0.5 * h;
            return 1;
        }
        lo[0] = 0.0;
        hi[0] = c + 0.5 * h;
        lo[1] = 0.0;
        hi[1] = 0.5 * h - c;
        return 2;
    }

    // Average of 1/r over a cell of size hx*hy*hz centered at (x,y,z), with F
    // evaluated only at non-negative arguments. This helper deliberately
    // returns the average without the 1/(4*pi) prefactor; the test applies
    // that prefactor at the same point where the solver does.
    KOKKOS_INLINE_FUNCTION double integrated_average(double x, double y, double z, double hx,
                                                     double hy, double hz) {
        double xlo[2], xhi[2], ylo[2], yhi[2], zlo[2], zhi[2];
        const int nx = nonnegative_segments(Kokkos::fabs(x), hx, xlo, xhi);
        const int ny = nonnegative_segments(Kokkos::fabs(y), hy, ylo, yhi);
        const int nz = nonnegative_segments(Kokkos::fabs(z), hz, zlo, zhi);

        double integral = 0.0;
        for (int a = 0; a < nx; ++a)
            for (int b = 0; b < ny; ++b)
                for (int c = 0; c < nz; ++c)
                    integral += box_integral(xlo[a], xhi[a], ylo[b], yhi[b], zlo[c], zhi[c]);

        return integral / (hx * hy * hz);
    }

    struct Errors {
        double unshifted;   // max |phi - ref| / max |ref|, unshifted kernel
        double shifted;     // same for shiftedGreensFunction(shift); 0 if not run
        double sourceCell;  // relative error in the source cell, unshifted kernel
    };

    // Solves for unit density in cell `source` on an N^3 mesh with spacing hr and
    // compares the potential with the reference cell average. If runShifted, the
    // same solver is then reused with shiftedGreensFunction(shift).
    Errors solve_and_compare(int N, const Vector_t& hr, const IndexVector& source, bool runShifted,
                             const Vector_t& shift) {
        ippl::NDIndex<Dim> owned;
        for (unsigned d = 0; d < Dim; ++d) {
            owned[d] = ippl::Index(N);
        }

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        Vector_t origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hr, origin);
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel);

        field_t rho, rho_clean, phi_unshifted, phi_shifted;
        rho.initialize(mesh, layout);
        rho_clean.initialize(mesh, layout);
        phi_unshifted.initialize(mesh, layout);
        phi_shifted.initialize(mesh, layout);

        // Put unit density in the source cell. The physical source charge is
        // therefore rho * cellVolume = cellVolume, which is why the expected
        // potential below carries an explicit cellVolume factor.
        {
            auto view        = rho.getView();
            auto clean       = rho_clean.getView();
            const int nghost = rho.getNghost();
            const auto& ldom = layout.getLocalNDIndex();
            Kokkos::parallel_for(
                "initialize unit cell density", rho.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;
                    const double value =
                        (ig == source[0] && jg == source[1] && kg == source[2]) ? 1.0 : 0.0;
                    view(i, j, k)  = value;
                    clean(i, j, k) = value;
                });
        }

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

        Solver_t solver(rho, params);
        solver.solve();
        Kokkos::deep_copy(phi_unshifted.getView(), rho.getView());

        if (runShifted) {
            Kokkos::deep_copy(rho.getView(), rho_clean.getView());
            solver.shiftedGreensFunction(shift);
            solver.solve();
            Kokkos::deep_copy(phi_shifted.getView(), rho.getView());
        }

        const double cellVolume = hr[0] * hr[1] * hr[2];
        const double inv4pi     = 1.0 / (4.0 * Kokkos::numbers::pi_v<double>);

        // Max abs error, max abs reference, and the relative error in the source
        // cell of a solved potential. kernelShift = 0 tests greensFunction();
        // kernelShift = shift tests shiftedGreensFunction(shift).
        auto compute_error = [&](field_t& phi, const Vector_t& kernelShift) {
            auto view        = phi.getView();
            const int nghost = phi.getNghost();
            const auto& ldom = layout.getLocalNDIndex();
            double local[3]  = {0.0, 0.0, 0.0};

            // solve() multiplies the inverse FFT result by the cell volume. The
            // integrated kernel itself is a cell average.
            auto expected_at = KOKKOS_LAMBDA(const int ig, const int jg, const int kg) {
                const double x = static_cast<double>(ig - source[0]) * hr[0] - kernelShift[0];
                const double y = static_cast<double>(jg - source[1]) * hr[1] - kernelShift[1];
                const double z = static_cast<double>(kg - source[2]) * hr[2] - kernelShift[2];
                return cellVolume * inv4pi * integrated_average(x, y, z, hr[0], hr[1], hr[2]);
            };

            Kokkos::parallel_reduce(
                "integrated green max error", phi.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& lmax) {
                    const int ig      = i + ldom[0].first() - nghost;
                    const int jg      = j + ldom[1].first() - nghost;
                    const int kg      = k + ldom[2].first() - nghost;
                    const double diff = Kokkos::fabs(view(i, j, k) - expected_at(ig, jg, kg));
                    if (diff > lmax)
                        lmax = diff;
                },
                Kokkos::Max<double>(local[0]));

            Kokkos::parallel_reduce(
                "integrated green max reference", phi.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& lmax) {
                    const int ig      = i + ldom[0].first() - nghost;
                    const int jg      = j + ldom[1].first() - nghost;
                    const int kg      = k + ldom[2].first() - nghost;
                    const double absv = Kokkos::fabs(expected_at(ig, jg, kg));
                    if (absv > lmax)
                        lmax = absv;
                },
                Kokkos::Max<double>(local[1]));

            Kokkos::parallel_reduce(
                "integrated green source cell error", phi.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& lmax) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;
                    if (ig != source[0] || jg != source[1] || kg != source[2])
                        return;
                    const double expected = expected_at(ig, jg, kg);
                    const double rel =
                        Kokkos::fabs(view(i, j, k) - expected) / Kokkos::fabs(expected);
                    if (rel > lmax)
                        lmax = rel;
                },
                Kokkos::Max<double>(local[2]));

            double global[3] = {0.0, 0.0, 0.0};
            ippl::Comm->allreduce(&local[0], &global[0], 3, std::greater<double>());
            return Errors{global[0] / global[1], 0.0, global[2]};
        };

        const Vector_t zeroShift = {0.0, 0.0, 0.0};
        Errors errors            = compute_error(phi_unshifted, zeroShift);
        if (runShifted) {
            errors.shifted = compute_error(phi_shifted, shift).unshifted;
        }
        return errors;
    }
}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int exit_code = 0;
    {
        Inform msg("TestIntegratedGreensFunction");

        const int N = 8;
        auto check  = [&](const char* label, double value, double tol) {
            msg << label << " = " << value << " (tolerance " << tol << ")" << endl;
            if (!(value <= tol)) {
                msg << "FAIL: " << label << " exceeds " << tol << endl;
                exit_code = 1;
            }
        };

        // Near-cubic mesh, source at the origin, unshifted and shifted kernels.
        // This is an exact closed-form kernel check up to FFT and floating-point
        // roundoff, so the tolerance can be much tighter than a physics
        // convergence test.
        {
            const Vector_t hr     = {0.17, 0.23, 0.31};
            const IndexVector src = {0, 0, 0};
            const Vector_t shift  = {0.41, -0.19, 0.37};
            const Errors e        = solve_and_compare(N, hr, src, true, shift);
            const double tol      = 5.0e-12;
            check("near-cubic unshifted relative error", e.unshifted, tol);
            check("near-cubic shifted relative error", e.shifted, tol);
        }

        // Elongated mesh, hz/hx = 1e5, source in the far corner: every kernel
        // offset is <= 0. The eight-corner difference itself loses about
        // eps * N^2 * (hz/hx) * ln(hz/hx) at far cells, so the global tolerance
        // is looser than above; the cancellation in log(a + r) at a < 0 gives
        // errors well above it. The source cell has no far-cell loss and checks
        // the half-cell corners at -h/2.
        {
            const Vector_t hr     = {1.0e-6, 1.0e-6, 0.1};
            const IndexVector src = {N - 1, N - 1, N - 1};
            const Vector_t shift  = {0.0, 0.0, 0.0};
            const Errors e        = solve_and_compare(N, hr, src, false, shift);
            check("elongated relative error", e.unshifted, 1.0e-8);
            check("elongated source cell relative error", e.sourceCell, 1.0e-11);
        }

        if (exit_code == 0) {
            msg << "PASS" << endl;
        }
    }
    ippl::finalize();
    return exit_code;
}
