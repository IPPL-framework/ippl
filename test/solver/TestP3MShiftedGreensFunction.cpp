/**
 * @file TestP3MShiftedGreensFunction.cpp
 * @brief Check the full P3M image kernel and restoration against point-source potentials.
 */

#include "Ippl.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "Utility/IpplException.h"

#include "PoissonSolvers/FFTTruncatedGreenPeriodicPoissonSolver.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int status = 0;
    {
        constexpr unsigned Dim = 3;
        using Mesh             = ippl::UniformCartesian<double, Dim>;
        using Vector           = ippl::Vector<double, Dim>;
        using Field            = ippl::Field<double, Dim, Mesh, Mesh::DefaultCentering>;
        using VectorField      = ippl::Field<Vector, Dim, Mesh, Mesh::DefaultCentering>;
        using Solver           = ippl::FFTTruncatedGreenPeriodicPoissonSolver<VectorField, Field>;
        ippl::NDIndex<Dim> domain(ippl::Index(8), ippl::Index(10), ippl::Index(12));
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, domain, {true, true, true}, false);
        Mesh mesh(domain, Vector(0.1, 0.12, 0.08), Vector(-0.4, -0.6, 0.2));
        Field rho(mesh, layout), reboundRho(mesh, layout);
        VectorField field(mesh, layout), reboundField(mesh, layout);
        const auto local = layout.getLocalNDIndex();
        const ippl::Vector<int, Dim> source(2, 3, 4);
        const double pi = Kokkos::numbers::pi_v<double>;

        auto deposit = [&](Field& rhs) {
            rhs       = 0.0;
            auto view = rhs.getHostMirror();
            Kokkos::deep_copy(view, rhs.getView());
            const int ghost = rhs.getNghost();
            if (local[0].first() <= source[0] && source[0] <= local[0].last()
                && local[1].first() <= source[1] && source[1] <= local[1].last()
                && local[2].first() <= source[2] && source[2] <= local[2].last()) {
                view(source[0] - local[0].first() + ghost, source[1] - local[1].first() + ghost,
                     source[2] - local[2].first() + ghost) = 1.0;
            }
            Kokkos::deep_copy(rhs.getView(), view);
        };

        auto verify = [&](Field& rhs, VectorField& electric, bool shifted, const Vector& shift,
                          double alpha, double coupling) {
            auto potential =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs.getView());
            auto e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), electric.getView());
            const int ghost     = rhs.getNghost();
            const Vector h      = mesh.getMeshSpacing();
            const double volume = h[0] * h[1] * h[2];
            const double hmin   = std::min({h[0], h[1], h[2]});
            double error = 0.0, norm = 0.0;
            int invalid = 0;
            for (int i = ghost; i < int(potential.extent(0)) - ghost; ++i) {
                for (int j = ghost; j < int(potential.extent(1)) - ghost; ++j) {
                    for (int k = ghost; k < int(potential.extent(2)) - ghost; ++k) {
                        Vector offset((i - ghost + local[0].first() - source[0]) * h[0],
                                      (j - ghost + local[1].first() - source[1]) * h[1],
                                      (k - ghost + local[2].first() - source[2]) * h[2]);
                        if (shifted)
                            offset -= shift;
                        const double r2 = offset.dot(offset);
                        const double r  = std::sqrt(r2);
                        const double kernel =
                            shifted
                                ? coupling
                                      / std::sqrt(
                                          r2 + (r2 < 0.25 * hmin * hmin ? 0.25 * hmin * hmin : 0.0))
                                : (r == 0.0 ? coupling * 2.0 * alpha / std::sqrt(pi)
                                            : coupling * std::erf(alpha * r) / r);
                        const double exact = -volume * kernel;
                        error += std::pow(potential(i, j, k) - exact, 2);
                        norm += exact * exact;
                        invalid += !std::isfinite(potential(i, j, k));
                        for (unsigned d = 0; d < Dim; ++d)
                            invalid += !std::isfinite(e(i, j, k)[d]);
                    }
                }
            }
            double globalError = 0.0, globalNorm = 0.0;
            int globalInvalid = 0;
            ippl::Comm->allreduce(error, globalError, 1, std::plus<double>());
            ippl::Comm->allreduce(norm, globalNorm, 1, std::plus<double>());
            ippl::Comm->allreduce(invalid, globalInvalid, 1, std::plus<int>());
            const double relativeError = std::sqrt(globalError / globalNorm);
            if (globalInvalid != 0 || !(relativeError < 1.0e-11)) {
                if (ippl::Comm->rank() == 0)
                    std::cerr << "P3M kernel mismatch: shifted=" << shifted << " alpha=" << alpha
                              << " coupling=" << coupling << " relative error=" << relativeError
                              << '\n';
                status = 1;
            }
        };

        Solver uninitialized;
        try {
            uninitialized.shiftedGreensFunction(Vector(0.0));
            status = 1;
        } catch (const IpplException&) {
        }

        for (double alpha : {0.8, 3.0}) {
            for (double coupling : {-1.0 / (4.0 * pi), -0.37, 0.29}) {
                ippl::ParameterList params;
                params.add("use_heffte_defaults", true);
                params.add("output_type", Solver::SOL_AND_GRAD);
                params.add("alpha", alpha);
                params.add("force_constant", coupling);
                params.add("boundary_type", Solver::OPEN);
                Solver solver(field, rho, params);
                for (double spacing : {0.1, 0.13}) {
                    // Install the overwrite before solve() can observe the changed spacing.
                    mesh.setMeshSpacing(Vector(spacing, 1.2 * spacing, 0.8 * spacing));
                    for (Vector shift : {Vector(0.11, -0.08, 0.63), Vector(0.0)}) {
                        deposit(rho);
                        solver.shiftedGreensFunction(shift);
                        solver.solve();
                        verify(rho, field, true, shift, alpha, coupling);
                        solver.greensFunction();
                        deposit(rho);
                        solver.solve();
                        verify(rho, field, false, Vector(0.0), alpha, coupling);
                    }
                }
                solver.shiftedGreensFunction(Vector(0.0, 0.0, 0.7));
                mesh.setMeshSpacing(Vector(0.12, 0.1, 0.09));
                deposit(rho);
                solver.solve();
                verify(rho, field, false, Vector(0.0), alpha, coupling);

                solver.shiftedGreensFunction(Vector(0.0, 0.0, 0.7));
                solver.setRhs(reboundRho);
                solver.setLhs(reboundField);
                deposit(reboundRho);
                solver.solve();
                verify(reboundRho, reboundField, false, Vector(0.0), alpha, coupling);
                deposit(reboundRho);
                const Vector shift(0.03, 0.01, 0.7);
                solver.shiftedGreensFunction(shift);
                solver.solve();
                verify(reboundRho, reboundField, true, shift, alpha, coupling);
            }
        }

        ippl::ParameterList periodicParams;
        periodicParams.add("use_heffte_defaults", true);
        periodicParams.add("boundary_type", Solver::PERIODIC);
        Solver periodic(field, rho, periodicParams);
        rho   = 2.0;
        field = Vector(3.0);
        try {
            periodic.shiftedGreensFunction(Vector(0.0, 0.0, 0.7));
            status = 1;
        } catch (const IpplException&) {
        }
        auto unchanged  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho.getView());
        auto unchangedE = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.getView());
        for (std::size_t i = 0; i < unchanged.extent(0); ++i)
            for (std::size_t j = 0; j < unchanged.extent(1); ++j)
                for (std::size_t k = 0; k < unchanged.extent(2); ++k) {
                    status |= unchanged(i, j, k) != 2.0;
                    for (unsigned d = 0; d < Dim; ++d)
                        status |= unchangedE(i, j, k)[d] != 3.0;
                }
        int globalStatus = 0;
        ippl::Comm->allreduce(status, globalStatus, 1, std::plus<int>());
        status = globalStatus == 0 ? 0 : 1;
    }
    ippl::finalize();
    return status;
}
