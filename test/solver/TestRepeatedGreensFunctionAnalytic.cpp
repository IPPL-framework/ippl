//
// TestRepeatedGreensFunctionAnalytic
//
// Analytic check of the `repeat_count` / `repeat_spacing` kernel copies of FFTOpenPoissonSolver.
// The source is the normalized Gaussian of TestGaussian_convergence (sigma 0.05, centred in the
// unit cube). By linearity, phi_K - phi_0 and E_K - E_0 are the field of the 2K copies alone, so
// the error of the central solve cancels, including the missing particle-particle part of
// TRUNCATED. Both differences are compared with the exact single-Gaussian field summed over the
// shifts k a e_z, 0 < |k| <= K. E comes from the spectral gradient on the doubled grid
// (SOL_AND_GRAD), which the potential-only checks of TestRepeatedGreensFunction do not reach.
// TRUNCATED runs with force_constant != -1/(4 pi), so the copies must scale with it.
//
//   1. Copies outside the box (a = 1 touching, a = 2.5 separated), K = 1 and 3, N = 48: the
//      copies' field is smooth in the box, so STANDARD and TRUNCATED reach round-off. INTEGRATED
//      reaches its O(h^4) cell-average error, plus round-off of the averages of far copies.
//      N = 48 is the coarsest grid on which the spectral gradient resolves the Gaussian to
//      round-off; its E error is about exp(-(pi sigma N)^2 / 2), 1e-13 at N = 48, 1e-6 at N = 32.
//   2. Copies overlapping the box (a = 0.37, not a multiple of h), K = 2: second-order
//      convergence from N = 32 to N = 48.
//
// Every case rebuilds the kernel on the doubled grid, and an INTEGRATED copy costs eight
// antiderivative evaluations per grid point, so grid size and K set the run time of the test.
//
// Usage: ./TestRepeatedGreensFunctionAnalytic
// Exit code: 0 on success, 1 on failure.
//

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cmath>
#include <sstream>
#include <string>

#include "PoissonSolvers/FFTOpenPoissonSolver.h"

namespace {
    constexpr unsigned Dim = 3;
    using Mesh_t           = ippl::UniformCartesian<double, Dim>;
    using Centering_t      = Mesh_t::DefaultCentering;
    using field_t          = ippl::Field<double, Dim, Mesh_t, Centering_t>;
    using vector_t         = ippl::Vector<double, Dim>;
    using fieldV_t         = ippl::Field<vector_t, Dim, Mesh_t, Centering_t>;
    using Solver_t         = ippl::FFTOpenPoissonSolver<fieldV_t, field_t>;

    constexpr double sigma = 0.05;
    constexpr double mu    = 0.5;

    KOKKOS_INLINE_FUNCTION double gaussian(double x, double y, double z) {
        const double pi = Kokkos::numbers::pi_v<double>;
        const double r2 = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);
        return Kokkos::exp(-r2 / (2 * sigma * sigma))
               / (Kokkos::sqrt(8 * pi * pi * pi) * sigma * sigma * sigma);
    }

    // Exact open-boundary solution of laplace(phi) = -rho for the Gaussian above.
    KOKKOS_INLINE_FUNCTION double exactPhi(double x, double y, double z) {
        const double pi = Kokkos::numbers::pi_v<double>;
        const double r =
            Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));
        return Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma)) / (4 * pi * r);
    }

    KOKKOS_INLINE_FUNCTION vector_t exactE(double x, double y, double z) {
        const double pi = Kokkos::numbers::pi_v<double>;
        const double r =
            Kokkos::sqrt((x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu));
        const double f =
            (Kokkos::erf(r / (Kokkos::sqrt(2.0) * sigma)) / r
             - Kokkos::sqrt(2.0 / pi) / sigma * Kokkos::exp(-r * r / (2 * sigma * sigma)))
            / (4 * pi * r * r);
        return vector_t{f * (x - mu), f * (y - mu), f * (z - mu)};
    }

    ippl::ParameterList solverParameters(int greens, double forceConstant) {
        ippl::ParameterList params;
        params.add("use_pencils", true);
        params.add("comm", ippl::a2a);
        params.add("use_reorder", false);
        params.add("use_heffte_defaults", false);
        params.add("use_gpu_aware", true);
        params.add("r2c_direction", 0);
        params.add("algorithm", Solver_t::HOCKNEY);
        params.add("output_type", Solver_t::SOL_AND_GRAD);
        params.add("greens_function", greens);
        params.add("alpha", 40.0);
        params.add("force_constant", forceConstant);
        params.add("repeat_count", 0);
        params.add("repeat_spacing", 0.0);
        return params;
    }

    void fillGaussian(field_t& rho, double h) {
        auto view        = rho.getView();
        const int nghost = rho.getNghost();
        const auto& ldom = rho.getLayout().getLocalNDIndex();
        Kokkos::parallel_for(
            "fill rho", rho.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                view(i, j, k) = gaussian((i + ldom[0].first() - nghost + 0.5) * h,
                                         (j + ldom[1].first() - nghost + 0.5) * h,
                                         (k + ldom[2].first() - nghost + 0.5) * h);
            });
    }

    // scale * sum over 0 < |k| <= K of the exact field shifted by k * a along z
    void fillCopies(field_t& phi, fieldV_t& E, double h, int K, double a, double scale) {
        auto vphi        = phi.getView();
        auto vE          = E.getView();
        const int nghost = phi.getNghost();
        const auto& ldom = phi.getLayout().getLocalNDIndex();
        Kokkos::parallel_for(
            "fill copies", phi.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const double x = (i + ldom[0].first() - nghost + 0.5) * h;
                const double y = (j + ldom[1].first() - nghost + 0.5) * h;
                const double z = (k + ldom[2].first() - nghost + 0.5) * h;
                double p       = 0;
                vector_t e(0.0);
                for (int c = -K; c <= K; ++c) {
                    if (c != 0) {
                        p += exactPhi(x, y, z - c * a);
                        e += exactE(x, y, z - c * a);
                    }
                }
                vphi(i, j, k) = scale * p;
                vE(i, j, k)   = scale * e;
            });
    }

    double sumSquares(fieldV_t& E) {
        auto view    = E.getView();
        double local = 0;
        Kokkos::parallel_reduce(
            "sum of squares", E.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum) {
                for (unsigned d = 0; d < Dim; ++d) {
                    sum += view(i, j, k)[d] * view(i, j, k)[d];
                }
            },
            Kokkos::Sum<double>(local));
        double global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
        return global;
    }

    struct Errors {
        double phi;
        double E;
    };

    bool report(Inform& msg, const std::string& name, double value, double bound,
                bool upper = true) {
        const bool ok = upper ? value < bound : value > bound;
        msg << (ok ? "PASS " : "FAIL ") << name << ": " << value << (upper ? " (max " : " (min ")
            << bound << ")" << endl;
        return ok;
    }
}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    bool ok = true;
    {
        Inform msg("TestRepeatedGreensFunctionAnalytic");
        const double pi = Kokkos::numbers::pi_v<double>;

        struct Kernel {
            int type;
            std::string name;
            double forceConstant;
            double tolPhi;  // copies outside the box, relative L2 error of phi_K - phi_0
            double tolE;    // same for E_K - E_0
        };
        const Kernel kernels[] = {
            {Solver_t::STANDARD, "STANDARD", -1.0 / (4 * pi), 1e-12, 1e-12},
            {Solver_t::INTEGRATED, "INTEGRATED", -1.0 / (4 * pi), 1e-8, 2e-7},
            {Solver_t::TRUNCATED, "TRUNCATED", -0.7 / (4 * pi), 1e-12, 1e-12}};
        const double overlapSpacing = 0.37;
        const int overlapCount      = 2;
        const int sizes[]           = {32, 48};
        const double sizeRatio      = static_cast<double>(sizes[1]) / sizes[0];

        for (const Kernel& kernel : kernels) {
            // TRUNCATED copies use force_constant as coupling; -1/(4 pi) solves laplace(phi) =
            // -rho.
            const double scale =
                kernel.type == Solver_t::TRUNCATED ? -4 * pi * kernel.forceConstant : 1.0;
            Errors overlap[2];

            for (int n = 0; n < 2; ++n) {
                const int N    = sizes[n];
                const double h = 1.0 / N;
                ippl::NDIndex<Dim> owned{ippl::Index(N), ippl::Index(N), ippl::Index(N)};
                std::array<bool, Dim> parallel;
                parallel.fill(true);
                Mesh_t mesh(owned, vector_t{h, h, h}, vector_t{0.0, 0.0, 0.0});
                ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, parallel);

                field_t rho, phi0, refPhi;
                fieldV_t E, E0, refE;
                for (field_t* f : {&rho, &phi0, &refPhi}) {
                    f->initialize(mesh, layout);
                }
                for (fieldV_t* f : {&E, &E0, &refE}) {
                    f->initialize(mesh, layout);
                }

                ippl::ParameterList params = solverParameters(kernel.type, kernel.forceConstant);
                fillGaussian(rho, h);
                Solver_t plain(E0, rho, params);
                plain.solve();
                Kokkos::deep_copy(phi0.getView(), rho.getView());

                // one solver for all cases; solve() regenerates the kernel when K or a change
                Solver_t repeated(E, rho, params);
                auto neighbourErrors = [&](int K, double a) {
                    repeated.updateParameter("repeat_count", K);
                    repeated.updateParameter("repeat_spacing", a);
                    fillGaussian(rho, h);
                    repeated.solve();
                    fillCopies(refPhi, refE, h, K, a, scale);
                    rho = rho - phi0 - refPhi;
                    E   = E - E0 - refE;
                    return Errors{norm(rho) / norm(refPhi),
                                  std::sqrt(sumSquares(E) / sumSquares(refE))};
                };

                if (N == sizes[1]) {
                    for (double a : {1.0, 2.5}) {
                        for (int K : {1, 3}) {
                            const Errors err = neighbourErrors(K, a);
                            std::ostringstream tag;
                            tag << kernel.name << " a=" << a << " K=" << K << " N=" << N << " ";
                            ok &= report(msg, tag.str() + "phi error", err.phi, kernel.tolPhi);
                            ok &= report(msg, tag.str() + "E error", err.E, kernel.tolE);
                        }
                    }
                }
                overlap[n] = neighbourErrors(overlapCount, overlapSpacing);
            }

            // Overlapping copies sample the kernel near its singularity, so their error is the
            // second-order near-field discretization error.
            const std::string tag = kernel.name + " overlapping copies ";
            ok &= report(msg, tag + "phi error at N=48", overlap[1].phi, 2e-3);
            ok &= report(msg, tag + "E error at N=48", overlap[1].E, 2e-2);
            ok &= report(msg, tag + "phi order",
                         std::log(overlap[0].phi / overlap[1].phi) / std::log(sizeRatio), 1.8,
                         false);
            ok &= report(msg, tag + "E order",
                         std::log(overlap[0].E / overlap[1].E) / std::log(sizeRatio), 1.8, false);
        }
    }
    ippl::finalize();
    return ok ? 0 : 1;
}
