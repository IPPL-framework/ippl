//
// TestFFTTruncatedGreenOpenPoissonSolver
// Validates that the P3M solver delegates OPEN boundary conditions to the
// doubled-grid Hockney solver with the Ewald long-range Green function.
//

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include "Utility/IpplException.h"

#include "PoissonSolvers/FFTTruncatedGreenPeriodicPoissonSolver.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int status = 0;
    {
        constexpr unsigned Dim = 3;
        using Mesh_t           = ippl::UniformCartesian<double, Dim>;
        using Centering_t      = Mesh_t::DefaultCentering;
        using Field_t          = ippl::Field<double, Dim, Mesh_t, Centering_t>;
        using Vector_t         = ippl::Vector<double, Dim>;
        using VField_t         = ippl::Field<Vector_t, Dim, Mesh_t, Centering_t>;
        using Solver_t         = ippl::FFTTruncatedGreenPeriodicPoissonSolver<VField_t, Field_t>;

        const ippl::Vector<int, Dim> nr          = {8, 8, 8};
        const ippl::Vector<int, Dim> sourceIndex = {3, 3, 3};
        ippl::NDIndex<Dim> owned;
        for (unsigned d = 0; d < Dim; ++d) {
            owned[d] = ippl::Index(nr[d]);
        }

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel, false);

        Vector_t hr(0.125);
        const Vector_t origin(-0.5);
        Mesh_t mesh(owned, hr, origin);
        Field_t rho(mesh, layout);
        VField_t efield(mesh, layout);

        constexpr double alpha         = 2.0;
        constexpr double forceConstant = -1.0 / (4.0 * Kokkos::numbers::pi_v<double>);

        ippl::ParameterList params;
        params.add("output_type", Solver_t::SOL_AND_GRAD);
        params.add("use_heffte_defaults", false);
        params.add("use_pencils", true);
        params.add("use_gpu_aware", true);
        params.add("comm", ippl::a2av);
        params.add("r2c_direction", 0);
        params.add("alpha", alpha);
        params.add("force_constant", forceConstant);
        params.add("boundary_type", Solver_t::OPEN);

        Solver_t solver(efield, rho, params);
        const auto& localDomain = layout.getLocalNDIndex();
        const int nghost        = rho.getNghost();

        auto assignPointSource = [&]() {
            rho          = 0.0;
            auto rhoView = rho.getView();
            Kokkos::parallel_for(
                "Assign open P3M point source", ippl::getRangePolicy(rhoView, nghost),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    const int ig = i + localDomain[0].first() - nghost;
                    const int jg = j + localDomain[1].first() - nghost;
                    const int kg = k + localDomain[2].first() - nghost;
                    if ((ig == sourceIndex[0]) && (jg == sourceIndex[1])
                        && (kg == sourceIndex[2])) {
                        rhoView(i, j, k) = 1.0;
                    }
                });
        };

        auto verifySolution = [&]() {
            const double cellVolume = hr[0] * hr[1] * hr[2];
            const double pi         = Kokkos::numbers::pi_v<double>;
            auto rhoView            = rho.getView();
            auto eView              = efield.getView();
            const int eNghost       = efield.getNghost();

            double errorLocal       = 0.0;
            double normLocal        = 0.0;
            int invalidLocal        = 0;
            double sourceFieldLocal = 0.0;
            Kokkos::parallel_reduce(
                "Open P3M direct convolution error", ippl::getRangePolicy(rhoView, nghost),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum) {
                    const int ig        = i + localDomain[0].first() - nghost;
                    const int jg        = j + localDomain[1].first() - nghost;
                    const int kg        = k + localDomain[2].first() - nghost;
                    const double dx     = (ig - sourceIndex[0]) * hr[0];
                    const double dy     = (jg - sourceIndex[1]) * hr[1];
                    const double dz     = (kg - sourceIndex[2]) * hr[2];
                    const double r2     = dx * dx + dy * dy + dz * dz;
                    const bool isOrigin = (r2 == 0.0);
                    const double r      = Kokkos::sqrt(r2);
                    const double safeR  = r + static_cast<double>(isOrigin);
                    const double green  = isOrigin ? forceConstant * 2.0 * alpha / Kokkos::sqrt(pi)
                                                   : forceConstant * Kokkos::erf(alpha * r) / safeR;
                    const double exact  = -cellVolume * green;
                    const double diff   = rhoView(i, j, k) - exact;
                    sum += diff * diff;
                },
                Kokkos::Sum<double>(errorLocal));
            Kokkos::parallel_reduce(
                "Open P3M direct convolution norm", ippl::getRangePolicy(rhoView, nghost),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum) {
                    const int ig        = i + localDomain[0].first() - nghost;
                    const int jg        = j + localDomain[1].first() - nghost;
                    const int kg        = k + localDomain[2].first() - nghost;
                    const double dx     = (ig - sourceIndex[0]) * hr[0];
                    const double dy     = (jg - sourceIndex[1]) * hr[1];
                    const double dz     = (kg - sourceIndex[2]) * hr[2];
                    const double r2     = dx * dx + dy * dy + dz * dz;
                    const bool isOrigin = (r2 == 0.0);
                    const double r      = Kokkos::sqrt(r2);
                    const double safeR  = r + static_cast<double>(isOrigin);
                    const double green  = isOrigin ? forceConstant * 2.0 * alpha / Kokkos::sqrt(pi)
                                                   : forceConstant * Kokkos::erf(alpha * r) / safeR;
                    const double exact  = -cellVolume * green;
                    sum += exact * exact;
                },
                Kokkos::Sum<double>(normLocal));
            Kokkos::parallel_reduce(
                "Open P3M finite field", ippl::getRangePolicy(eView, eNghost),
                KOKKOS_LAMBDA(const int i, const int j, const int k, int& count) {
                    for (unsigned d = 0; d < Dim; ++d) {
                        count += !Kokkos::isfinite(eView(i, j, k)[d]);
                    }
                },
                Kokkos::Sum<int>(invalidLocal));
            Kokkos::parallel_reduce(
                "Open P3M source field", ippl::getRangePolicy(eView, eNghost),
                KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum) {
                    const int ig = i + localDomain[0].first() - eNghost;
                    const int jg = j + localDomain[1].first() - eNghost;
                    const int kg = k + localDomain[2].first() - eNghost;
                    if ((ig == sourceIndex[0]) && (jg == sourceIndex[1])
                        && (kg == sourceIndex[2])) {
                        sum += eView(i, j, k).dot(eView(i, j, k));
                    }
                },
                Kokkos::Sum<double>(sourceFieldLocal));

            double error       = 0.0;
            double norm        = 0.0;
            int invalid        = 0;
            double sourceField = 0.0;
            ippl::Comm->allreduce(errorLocal, error, 1, std::plus<double>());
            ippl::Comm->allreduce(normLocal, norm, 1, std::plus<double>());
            ippl::Comm->allreduce(invalidLocal, invalid, 1, std::plus<int>());
            ippl::Comm->allreduce(sourceFieldLocal, sourceField, 1, std::plus<double>());

            const double relativeError = std::sqrt(error / norm);
            bool oddSymmetry           = true;
            double sampledFieldNorm    = 0.0;
            for (unsigned axis = 0; axis < Dim; ++axis) {
                auto plusIndex  = sourceIndex;
                auto minusIndex = sourceIndex;
                ++plusIndex[axis];
                --minusIndex[axis];

                for (unsigned component = 0; component < Dim; ++component) {
                    double plusLocal  = 0.0;
                    double minusLocal = 0.0;
                    Kokkos::parallel_reduce(
                        "Open P3M positive symmetry sample", ippl::getRangePolicy(eView, eNghost),
                        KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum) {
                            const int ig = i + localDomain[0].first() - eNghost;
                            const int jg = j + localDomain[1].first() - eNghost;
                            const int kg = k + localDomain[2].first() - eNghost;
                            if ((ig == plusIndex[0]) && (jg == plusIndex[1])
                                && (kg == plusIndex[2])) {
                                sum += eView(i, j, k)[component];
                            }
                        },
                        Kokkos::Sum<double>(plusLocal));
                    Kokkos::parallel_reduce(
                        "Open P3M negative symmetry sample", ippl::getRangePolicy(eView, eNghost),
                        KOKKOS_LAMBDA(const int i, const int j, const int k, double& sum) {
                            const int ig = i + localDomain[0].first() - eNghost;
                            const int jg = j + localDomain[1].first() - eNghost;
                            const int kg = k + localDomain[2].first() - eNghost;
                            if ((ig == minusIndex[0]) && (jg == minusIndex[1])
                                && (kg == minusIndex[2])) {
                                sum += eView(i, j, k)[component];
                            }
                        },
                        Kokkos::Sum<double>(minusLocal));

                    double plus  = 0.0;
                    double minus = 0.0;
                    ippl::Comm->allreduce(plusLocal, plus, 1, std::plus<double>());
                    ippl::Comm->allreduce(minusLocal, minus, 1, std::plus<double>());
                    oddSymmetry = oddSymmetry && std::abs(plus + minus) < 1.0e-10;
                    sampledFieldNorm += plus * plus + minus * minus;
                }
            }
            if (ippl::Comm->rank() == 0) {
                std::cout << "Open P3M relative potential error: " << relativeError << std::endl;
            }
            return relativeError < 1.0e-10 && invalid == 0 && std::sqrt(sourceField) < 1.0e-10
                   && oddSymmetry && sampledFieldNorm > 1.0e-20;
        };

        assignPointSource();
        solver.solve();
        status |= !verifySolution();

        // A spacing change must rebuild the same truncated kernel, not the standard 1/r kernel.
        hr = Vector_t(0.1);
        mesh.setMeshSpacing(hr);
        assignPointSource();
        solver.solve();
        status |= !verifySolution();

        ippl::ParameterList invalidBoundaryParams = params;
        invalidBoundaryParams.update("boundary_type", 7);
        try {
            Solver_t invalidBoundarySolver(efield, rho, invalidBoundaryParams);
            status = 1;
        } catch (const IpplException&) {
        }

        ippl::ParameterList invalidAlphaParams = params;
        invalidAlphaParams.update("alpha", 0.0);
        try {
            Solver_t invalidAlphaSolver(efield, rho, invalidAlphaParams);
            status = 1;
        } catch (const IpplException&) {
        }

        int globalStatus = 0;
        ippl::Comm->allreduce(status, globalStatus, 1, std::plus<int>());
        status = globalStatus == 0 ? 0 : 1;
    }
    ippl::finalize();
    return status;
}
