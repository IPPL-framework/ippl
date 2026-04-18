//
// TestShiftedGreensFunction
//
// Validates FFTOpenPoissonSolver::shiftedGreensFunction. A Gaussian charge is
// placed off-center inside a cubic box whose lower face (z = 0) is a Dirichlet
// plane. The caller orchestrates the image correction externally:
//
//   1. solve() with the standard Green's function            -> phi_open
//   2. shiftedGreensFunction(shift) + solve()                  -> phi_raw
//   3. axis-flip phi_raw in z and negate                       -> phi_image
//   4. phi_total = phi_open + phi_image
//   5. greensFunction() restores the cached kernel
//
// With shift_z = 2 * (plane_z - z_domain_center), this enforces phi(plane) ~= 0
// to within discretization error and reproduces the analytical image-dipole
// solution in the bulk.
//
// Usage (single rank):
//     ./TestShiftedGreensFunction
//
// Exit code: 0 on success, 1 on failure.
//

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <cmath>
#include <cstdlib>
#include <functional>

#include "PoissonSolvers/FFTOpenPoissonSolver.h"

namespace {
    // Gaussian charge density centered at mu with width sigma; integrated charge = 1.
    KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z,
                                           double mu_x, double mu_y, double mu_z,
                                           double sigma) {
        const double pi  = Kokkos::numbers::pi_v<double>;
        const double s3  = sigma * sigma * sigma;
        const double pre = 1.0 / (Kokkos::sqrt(8.0 * pi * pi * pi) * s3);
        const double dx  = x - mu_x;
        const double dy  = y - mu_y;
        const double dz  = z - mu_z;
        const double r2  = dx * dx + dy * dy + dz * dz;
        return pre * Kokkos::exp(-r2 / (2.0 * sigma * sigma));
    }

    // Free-space potential of a unit-charge Gaussian at mu.
    KOKKOS_INLINE_FUNCTION double gaussian_potential(double x, double y, double z,
                                                     double mu_x, double mu_y, double mu_z,
                                                     double sigma) {
        const double pi = Kokkos::numbers::pi_v<double>;
        const double dx = x - mu_x;
        const double dy = y - mu_y;
        const double dz = z - mu_z;
        const double r  = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1e-12) {
            return 1.0 / (4.0 * pi * Kokkos::sqrt(2.0 * pi) * sigma);
        }
        return (1.0 / (4.0 * pi * r)) * Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma));
    }
}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int exit_code = 0;
    {
        Inform msg("TestShiftedGreensFunction");

        constexpr unsigned int Dim = 3;
        using Mesh_t      = ippl::UniformCartesian<double, Dim>;
        using Centering_t = Mesh_t::DefaultCentering;
        using field_t     = ippl::Field<double, Dim, Mesh_t, Centering_t>;
        using fieldV_t    = ippl::Field<ippl::Vector<double, Dim>, Dim, Mesh_t, Centering_t>;
        using Solver_t    = ippl::FFTOpenPoissonSolver<fieldV_t, field_t>;

        // Geometry: box [0, L]^3, Dirichlet plane at z = 0, charge center at
        // (L/2, L/2, L/4). The image charge is at (L/2, L/2, -L/4), entirely
        // outside the mesh — exactly the scenario the shifted Green's function
        // handles correctly (and the physical-image-particle approach does not).
        const int N          = 32;
        const double L       = 1.0;
        const double sigma   = 0.05;
        const double mu_x    = 0.5 * L;
        const double mu_y    = 0.5 * L;
        const double mu_z    = 0.25 * L;
        const double plane_z = 0.0;

        ippl::NDIndex<Dim> owned;
        for (unsigned i = 0; i < Dim; ++i) {
            owned[i] = ippl::Index(N);
        }
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        ippl::Vector<double, Dim> hr     = {L / N, L / N, L / N};
        ippl::Vector<double, Dim> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hr, origin);
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel);

        field_t rho, rho_clean, phi_open, phi_image, phi_total;
        rho.initialize(mesh, layout);
        rho_clean.initialize(mesh, layout);
        phi_open.initialize(mesh, layout);
        phi_image.initialize(mesh, layout);
        phi_total.initialize(mesh, layout);

        // Fill rho (and a stash copy rho_clean) with the Gaussian.
        {
            auto view        = rho.getView();
            auto view_clean  = rho_clean.getView();
            const int nghost = rho.getNghost();
            const auto& ldom = layout.getLocalNDIndex();
            Kokkos::parallel_for(
                "init rho", rho.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;
                    const double x = (ig + 0.5) * hr[0] + origin[0];
                    const double y = (jg + 0.5) * hr[1] + origin[1];
                    const double z = (kg + 0.5) * hr[2] + origin[2];
                    const double g = gaussian(x, y, z, mu_x, mu_y, mu_z, sigma);
                    view(i, j, k)       = g;
                    view_clean(i, j, k) = g;
                });
        }

        // Solver parameters. Output type = SOL so the potential ends up in RHS.
        ippl::ParameterList params;
        params.add("use_pencils", true);
        params.add("comm", ippl::a2a);
        params.add("use_reorder", false);
        params.add("use_heffte_defaults", false);
        params.add("use_gpu_aware", true);
        params.add("r2c_direction", 0);
        params.add("algorithm", Solver_t::HOCKNEY);
        params.add("output_type", Solver_t::SOL);

        Solver_t solver(rho, params);

        // (1) Open-BC solve with the standard Green's function cached from
        //     initializeFields(). rho is overwritten with phi_open.
        solver.solve();
        Kokkos::deep_copy(phi_open.getView(), rho.getView());

        // Restore rho to the charge density before the image correction solve.
        Kokkos::deep_copy(rho.getView(), rho_clean.getView());

        // (2) Install the shifted Green's function (Dirichlet at z = plane_z)
        //     and solve again. rho is overwritten with conv(rho, G_shifted).
        //
        // Shift derivation: cell-centered fields sample z_k = (k+0.5)*hz + origin.
        // Z-flipping (k -> N-1-k) maps z_k -> L + 2*origin - z_k. After the
        // shifted convolution G_shifted(r) = G0(r - shift), the post-flip z
        // argument of G0 is (L + 2*origin - z - z_src - shift_z). Setting its
        // magnitude equal to |z + z_src - 2*plane_z| (the image distance)
        // gives shift_z = L + 2*origin_z - 2*plane_z = 2*(z_center - plane_z).
        const double z_center = origin[2] + 0.5 * L;
        ippl::Vector<double, Dim> shift = {0.0, 0.0, 2.0 * (z_center - plane_z)};
        solver.shiftedGreensFunction(shift);
        solver.solve();
        // rho now holds the raw shifted convolution.

        // (3) Axis-flip in z and negate to obtain phi_image.
        //     (negate = image charge has opposite sign; flip = image position).
        {
            auto src    = rho.getView();
            auto dst    = phi_image.getView();
            const int gs = rho.getNghost();
            const int gd = phi_image.getNghost();
            // Assume identical N in all axes and single-rank (checked by solver
            // upstream via Comm->size() > 1 guard when we later add multi-rank).
            Kokkos::parallel_for(
                "z-flip + negate", phi_image.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    // Physical k -> flipped k_src = N-1-k; adjust for nghost
                    // in both views.
                    const int kg_dst = k - gd;           // 0..N-1 in physical
                    const int k_src  = (N - 1 - kg_dst) + gs;
                    const int i_src  = (i - gd) + gs;
                    const int j_src  = (j - gd) + gs;
                    dst(i, j, k)     = -src(i_src, j_src, k_src);
                });
        }

        // (4) Restore the cached kernel so future solve() calls use the
        //     standard free-space Green's function again.
        solver.greensFunction();

        // (5) Compose the total: phi_total = phi_open + phi_image.
        {
            auto v_open  = phi_open.getView();
            auto v_img   = phi_image.getView();
            auto v_total = phi_total.getView();
            Kokkos::parallel_for(
                "sum", phi_total.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    v_total(i, j, k) = v_open(i, j, k) + v_img(i, j, k);
                });
        }

        // Diagnostic 1: phi_total on the Dirichlet slab (k = 0, physical cells
        // at z = 0.5 * hz) relative to the bulk maximum.
        double maxOnPlane = 0.0;
        double maxBulk    = 0.0;
        {
            auto view        = phi_total.getView();
            const int nghost = phi_total.getNghost();
            const auto& ldom = layout.getLocalNDIndex();
            double localPlane = 0.0, localBulk = 0.0;

            Kokkos::parallel_reduce(
                "max on plane", phi_total.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& lmax) {
                    const int kg = k + ldom[2].first() - nghost;
                    const double absv = Kokkos::fabs(view(i, j, k));
                    if (kg == 0 && absv > lmax) lmax = absv;
                },
                Kokkos::Max<double>(localPlane));

            Kokkos::parallel_reduce(
                "max bulk", phi_total.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& lmax) {
                    const double absv = Kokkos::fabs(view(i, j, k));
                    if (absv > lmax) lmax = absv;
                },
                Kokkos::Max<double>(localBulk));

            ippl::Comm->allreduce(localPlane, maxOnPlane, 1, std::greater<double>());
            ippl::Comm->allreduce(localBulk, maxBulk, 1, std::greater<double>());
        }

        // Diagnostic 2: relative L2 error vs analytical image dipole, evaluated
        // outside the 3-sigma Gaussian core to avoid the kernel smoothing bias.
        double l2err = 0.0, l2ref = 0.0;
        {
            auto view        = phi_total.getView();
            const int nghost = phi_total.getNghost();
            const auto& ldom = layout.getLocalNDIndex();
            double localErr = 0.0, localRef = 0.0;

            Kokkos::parallel_reduce(
                "L2 err", phi_total.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& le) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;
                    const double x = (ig + 0.5) * hr[0] + origin[0];
                    const double y = (jg + 0.5) * hr[1] + origin[1];
                    const double z = (kg + 0.5) * hr[2] + origin[2];
                    const double dx = x - mu_x, dy = y - mu_y, dz = z - mu_z;
                    if (dx*dx + dy*dy + dz*dz < 9.0 * sigma * sigma) return;
                    const double phi_real  = gaussian_potential(x, y, z, mu_x, mu_y, mu_z, sigma);
                    const double phi_img   = gaussian_potential(x, y, z, mu_x, mu_y,
                                                                2.0 * plane_z - mu_z, sigma);
                    const double phi_exact = phi_real - phi_img;
                    const double diff      = view(i, j, k) - phi_exact;
                    le += diff * diff;
                },
                Kokkos::Sum<double>(localErr));

            Kokkos::parallel_reduce(
                "L2 ref", phi_total.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& lr) {
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;
                    const double x = (ig + 0.5) * hr[0] + origin[0];
                    const double y = (jg + 0.5) * hr[1] + origin[1];
                    const double z = (kg + 0.5) * hr[2] + origin[2];
                    const double dx = x - mu_x, dy = y - mu_y, dz = z - mu_z;
                    if (dx*dx + dy*dy + dz*dz < 9.0 * sigma * sigma) return;
                    const double phi_real  = gaussian_potential(x, y, z, mu_x, mu_y, mu_z, sigma);
                    const double phi_img   = gaussian_potential(x, y, z, mu_x, mu_y,
                                                                2.0 * plane_z - mu_z, sigma);
                    const double phi_exact = phi_real - phi_img;
                    lr += phi_exact * phi_exact;
                },
                Kokkos::Sum<double>(localRef));

            double gErr = 0.0, gRef = 0.0;
            ippl::Comm->allreduce(localErr, gErr, 1, std::plus<double>());
            ippl::Comm->allreduce(localRef, gRef, 1, std::plus<double>());
            l2err = std::sqrt(gErr);
            l2ref = std::sqrt(gRef);
        }

        const double relErr = (l2ref > 0) ? l2err / l2ref : 0.0;
        const double ratio  = (maxBulk > 0) ? maxOnPlane / maxBulk : 0.0;

        msg << "grid = " << N << "^3, L = " << L << ", sigma = " << sigma << endl;
        msg << "charge at (" << mu_x << ", " << mu_y << ", " << mu_z
            << "); plane z = " << plane_z << ", shift_z = " << shift[2] << endl;
        msg << "max|phi| on plane / max|phi| bulk = " << maxOnPlane << " / "
            << maxBulk << " = " << ratio << endl;
        msg << "rel L2 error vs analytical image dipole (r > 3 sigma) = "
            << relErr << endl;

        // Generous thresholds for first-cut verification; they detect sign /
        // flip / shift errors but leave room for O(h) bias near the plane.
        const double planeTol = 0.05;
        const double l2Tol    = 0.10;

        if (maxBulk <= 0.0) {
            msg << "FAIL: phi_total is identically zero." << endl;
            exit_code = 1;
        } else if (ratio > planeTol) {
            msg << "FAIL: phi on Dirichlet plane exceeds " << planeTol
                << " * bulk max." << endl;
            exit_code = 1;
        } else if (relErr > l2Tol) {
            msg << "FAIL: relative L2 error exceeds " << l2Tol << "." << endl;
            exit_code = 1;
        } else {
            msg << "PASS" << endl;
        }
    }
    ippl::finalize();
    return exit_code;
}
