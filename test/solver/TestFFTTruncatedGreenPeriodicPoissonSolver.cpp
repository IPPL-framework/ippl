#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cmath>
#include <iostream>

#include "PoissonSolvers/FFTTruncatedGreenPeriodicPoissonSolver.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int status = 0;
    {
        constexpr unsigned dim = 3;
        using Mesh_t           = ippl::UniformCartesian<double, dim>;
        using Centering_t      = Mesh_t::DefaultCentering;
        using Field_t          = ippl::Field<double, dim, Mesh_t, Centering_t>;
        using Vector_t         = ippl::Vector<double, dim>;
        using VField_t         = ippl::Field<Vector_t, dim, Mesh_t, Centering_t>;
        using Solver_t         = ippl::FFTTruncatedGreenPeriodicPoissonSolver<VField_t, Field_t>;

        ippl::Vector<int, dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};
        ippl::NDIndex<dim> owned;
        for (unsigned d = 0; d < dim; ++d) {
            owned[d] = ippl::Index(nr[d]);
        }

        std::array<bool, dim> isParallel;
        isParallel.fill(true);
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        Vector_t hr     = {1.0 / nr[0], 1.0 / nr[1], 1.0 / nr[2]};
        Vector_t origin = {-0.5, -0.5, -0.5};
        Mesh_t mesh(owned, hr, origin);
        Field_t rho(mesh, layout);
        VField_t efield(mesh, layout);

        constexpr double alpha  = 2.0;
        const double pi         = Kokkos::numbers::pi_v<double>;
        const double waveNumber = 2.0 * pi;
        const double screening  = std::exp(-waveNumber * waveNumber / (4.0 * alpha * alpha));
        const auto& localDomain = layout.getLocalNDIndex();
        const int nghost        = rho.getNghost();
        auto rhoView            = rho.getView();

        Kokkos::parallel_for(
            "Assign P3M periodic mode", ippl::getRangePolicy(rhoView, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int ig     = i + localDomain[0].first() - nghost;
                const double x   = origin[0] + (ig + 0.5) * hr[0];
                rhoView(i, j, k) = Kokkos::sin(waveNumber * x);
            });

        ippl::ParameterList params;
        params.add("output_type", Solver_t::SOL_AND_GRAD);
        params.add("use_heffte_defaults", false);
        params.add("use_pencils", true);
        params.add("use_gpu_aware", true);
        params.add("comm", ippl::a2av);
        params.add("r2c_direction", 0);
        params.add("alpha", alpha);
        params.add("force_constant", -1.0 / (4.0 * pi));

        Solver_t solver(efield, rho, params);
        solver.solve();

        auto eView           = efield.getView();
        double phiErrorLocal = 0.0;
        double phiNormLocal  = 0.0;
        double eErrorLocal   = 0.0;
        double eNormLocal    = 0.0;

        Kokkos::parallel_reduce(
            "P3M periodic potential error", ippl::getRangePolicy(rhoView, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum) {
                const int ig   = i + localDomain[0].first() - nghost;
                const double x = origin[0] + (ig + 0.5) * hr[0];
                const double exact =
                    screening * Kokkos::sin(waveNumber * x) / (waveNumber * waveNumber);
                const double diff = rhoView(i, j, k) - exact;
                sum += diff * diff;
            },
            Kokkos::Sum<double>(phiErrorLocal));
        Kokkos::parallel_reduce(
            "P3M periodic potential norm", ippl::getRangePolicy(rhoView, nghost),
            KOKKOS_LAMBDA(const int i, const int, const int, double& sum) {
                const int ig   = i + localDomain[0].first() - nghost;
                const double x = origin[0] + (ig + 0.5) * hr[0];
                const double exact =
                    screening * Kokkos::sin(waveNumber * x) / (waveNumber * waveNumber);
                sum += exact * exact;
            },
            Kokkos::Sum<double>(phiNormLocal));
        Kokkos::parallel_reduce(
            "P3M periodic field error", ippl::getRangePolicy(rhoView, nghost),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum) {
                const int ig       = i + localDomain[0].first() - nghost;
                const double x     = origin[0] + (ig + 0.5) * hr[0];
                const double exact = -screening * Kokkos::cos(waveNumber * x) / waveNumber;
                const double dx    = eView(i, j, k)[0] - exact;
                sum += dx * dx + eView(i, j, k)[1] * eView(i, j, k)[1]
                       + eView(i, j, k)[2] * eView(i, j, k)[2];
            },
            Kokkos::Sum<double>(eErrorLocal));
        Kokkos::parallel_reduce(
            "P3M periodic field norm", ippl::getRangePolicy(rhoView, nghost),
            KOKKOS_LAMBDA(const int i, const int, const int, double& sum) {
                const int ig       = i + localDomain[0].first() - nghost;
                const double x     = origin[0] + (ig + 0.5) * hr[0];
                const double exact = -screening * Kokkos::cos(waveNumber * x) / waveNumber;
                sum += exact * exact;
            },
            Kokkos::Sum<double>(eNormLocal));

        double phiError = 0.0;
        double phiNorm  = 0.0;
        double eError   = 0.0;
        double eNorm    = 0.0;
        ippl::Comm->allreduce(phiErrorLocal, phiError, 1, std::plus<double>());
        ippl::Comm->allreduce(phiNormLocal, phiNorm, 1, std::plus<double>());
        ippl::Comm->allreduce(eErrorLocal, eError, 1, std::plus<double>());
        ippl::Comm->allreduce(eNormLocal, eNorm, 1, std::plus<double>());

        const double relativePhiError = std::sqrt(phiError / phiNorm);
        const double relativeEError   = std::sqrt(eError / eNorm);
        if (ippl::Comm->rank() == 0) {
            std::cout << "Relative potential error: " << relativePhiError << '\n'
                      << "Relative field error: " << relativeEError << std::endl;
        }
        status = (relativePhiError < 1.0e-12 && relativeEError < 1.0e-12) ? 0 : 1;
    }
    ippl::finalize();
    return status;
}
