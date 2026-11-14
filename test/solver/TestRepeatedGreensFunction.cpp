//
// TestRepeatedGreensFunction
//
// Validates the `repeat_count` / `repeat_spacing` parameters of FFTOpenPoissonSolver, which add
// 2K copies of the kernel shifted by k * a along z (0 < |k| <= K).
//
//   1. Linearity, for STANDARD, INTEGRATED and TRUNCATED: one solve with the repeated kernel
//      equals the plain solve plus 2K solves with shiftedGreensFunction(k a e_z). The spacing is
//      not a multiple of h_z and the source is longer than a, so the shifted singularities fall
//      inside the doubled grid. Changing the parameters regenerates the kernel.
//   2. Explicit train (single rank), for STANDARD and INTEGRATED: with a equal to the box
//      length, the repeated solve on N^3 equals the solve of the same charge tiled 2K + 1 times
//      on an N x N x (2K + 1) N mesh, restricted to the central block.
//
// Usage: ./TestRepeatedGreensFunction
// Exit code: 0 on success, 1 on failure.
//

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>

#include <cmath>
#include <string>

#include "PoissonSolvers/FFTOpenPoissonSolver.h"

namespace {
    constexpr unsigned Dim = 3;
    using Mesh_t           = ippl::UniformCartesian<double, Dim>;
    using Centering_t      = Mesh_t::DefaultCentering;
    using field_t          = ippl::Field<double, Dim, Mesh_t, Centering_t>;
    using fieldV_t         = ippl::Field<ippl::Vector<double, Dim>, Dim, Mesh_t, Centering_t>;
    using Solver_t         = ippl::FFTOpenPoissonSolver<fieldV_t, field_t>;

    ippl::ParameterList solverParameters(int greens) {
        ippl::ParameterList params;
        params.add("use_pencils", true);
        params.add("comm", ippl::a2a);
        params.add("use_reorder", false);
        params.add("use_heffte_defaults", false);
        params.add("use_gpu_aware", true);
        params.add("r2c_direction", 0);
        params.add("algorithm", Solver_t::HOCKNEY);
        params.add("output_type", Solver_t::SOL);
        params.add("greens_function", greens);
        params.add("alpha", 40.0);
        params.add("force_constant", -1.0 / (4.0 * Kokkos::numbers::pi_v<double>));
        return params;
    }

    // Off-centre Gaussian, elongated along z; `period` > 0 tiles the block of length Lz.
    void fillCharge(field_t& rho, const ippl::Vector<double, Dim>& hr, double sigmaZ) {
        auto view        = rho.getView();
        const int nghost = rho.getNghost();
        const auto& ldom = rho.getLayout().getLocalNDIndex();
        Kokkos::parallel_for(
            "fill rho", rho.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const double x  = (i + ldom[0].first() - nghost + 0.5) * hr[0] - 0.43;
                const double y  = (j + ldom[1].first() - nghost + 0.5) * hr[1] - 0.55;
                const double z  = (k + ldom[2].first() - nghost + 0.5) * hr[2] - 0.47;
                view(i, j, k) = Kokkos::exp(-(x * x + y * y) / (2 * 0.08 * 0.08)
                                            - z * z / (2 * sigmaZ * sigmaZ));
            });
    }

    // max |a - b| / max |b| over the owned cells of all ranks
    double relativeDifference(field_t& a, field_t& b) {
        auto va          = a.getView();
        auto vb          = b.getView();
        const int nghost = a.getNghost();
        using mdrange    = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        double local[2]  = {0, 0};
        Kokkos::parallel_reduce(
            "relative difference",
            mdrange({nghost, nghost, nghost},
                    {va.extent(0) - nghost, va.extent(1) - nghost, va.extent(2) - nghost}),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& diff, double& ref) {
                diff = Kokkos::max(diff, Kokkos::abs(va(i, j, k) - vb(i, j, k)));
                ref  = Kokkos::max(ref, Kokkos::abs(vb(i, j, k)));
            },
            Kokkos::Max<double>(local[0]), Kokkos::Max<double>(local[1]));
        double global[2];
        MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, ippl::Comm->getCommunicator());
        return global[0] / global[1];
    }

    bool report(Inform& msg, const std::string& name, double err, double tol) {
        const bool ok = err < tol;
        msg << (ok ? "PASS " : "FAIL ") << name << ": relative max difference " << err
            << " (tolerance " << tol << ")" << endl;
        return ok;
    }
}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    bool ok = true;
    {
        Inform msg("TestRepeatedGreensFunction");
        const int N   = 16;
        const int K   = 2;
        const double L = 1.0;

        ippl::NDIndex<Dim> owned;
        for (unsigned d = 0; d < Dim; ++d) {
            owned[d] = ippl::Index(N);
        }
        std::array<bool, Dim> parallel;
        parallel.fill(true);
        // anisotropic cells: the box is twice as long in z
        ippl::Vector<double, Dim> hr     = {L / N, L / N, 2 * L / N};
        ippl::Vector<double, Dim> origin = {0.0, 0.0, 0.0};
        Mesh_t mesh(owned, hr, origin);
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, parallel);

        field_t rho, phiRepeated, phiReference, work;
        for (field_t* f : {&rho, &phiRepeated, &phiReference, &work}) {
            f->initialize(mesh, layout);
        }

        // 1. Linearity against shiftedGreensFunction for every kernel type
        const double spacing = 0.37 * 2 * L;  // not a multiple of h_z, shorter than the source
        const std::pair<int, std::string> kernels[] = {{Solver_t::STANDARD, "STANDARD"},
                                                       {Solver_t::INTEGRATED, "INTEGRATED"},
                                                       {Solver_t::TRUNCATED, "TRUNCATED"}};
        for (const auto& [greens, name] : kernels) {
            ippl::ParameterList params = solverParameters(greens);
            params.add("repeat_count", K);
            params.add("repeat_spacing", spacing);
            fillCharge(rho, hr, 0.6);
            Solver_t repeated(rho, params);
            repeated.solve();
            Kokkos::deep_copy(phiRepeated.getView(), rho.getView());

            fillCharge(rho, hr, 0.6);
            ippl::ParameterList plain = solverParameters(greens);
            Solver_t single(rho, plain);
            single.solve();
            Kokkos::deep_copy(phiReference.getView(), rho.getView());
            for (int k = -K; k <= K; ++k) {
                if (k == 0) {
                    continue;
                }
                fillCharge(rho, hr, 0.6);
                single.shiftedGreensFunction(ippl::Vector<double, Dim>{0.0, 0.0, k * spacing});
                single.solve();
                phiReference = phiReference + rho;
            }
            ok &= report(msg, name + " repeated kernel vs shifted solves",
                         relativeDifference(phiRepeated, phiReference), 1e-12);

            // Switching the repeats off must regenerate the plain kernel.
            fillCharge(rho, hr, 0.6);
            single.greensFunction();
            single.solve();
            Kokkos::deep_copy(phiReference.getView(), rho.getView());
            fillCharge(rho, hr, 0.6);
            repeated.updateParameter("repeat_count", 0);
            repeated.solve();
            ok &= report(msg, name + " repeat_count 0 after repeats",
                         relativeDifference(rho, phiReference), 1e-12);
        }

        // 2. Explicit train of 2K + 1 tiled blocks (single rank, needs a direct cell mapping)
        if (ippl::Comm->size() == 1) {
            ippl::NDIndex<Dim> ownedLong;
            ownedLong[0] = ippl::Index(N);
            ownedLong[1] = ippl::Index(N);
            ownedLong[2] = ippl::Index((2 * K + 1) * N);
            Mesh_t meshLong(ownedLong, hr, origin);
            ippl::FieldLayout<Dim> layoutLong(MPI_COMM_WORLD, ownedLong, parallel);
            field_t rhoLong;
            rhoLong.initialize(meshLong, layoutLong);

            for (const auto& [greens, name] : {kernels[0], kernels[1]}) {
                ippl::ParameterList params = solverParameters(greens);
                params.add("repeat_count", K);
                params.add("repeat_spacing", N * hr[2]);
                fillCharge(rho, hr, 0.3);
                Kokkos::deep_copy(work.getView(), rho.getView());
                Solver_t repeated(rho, params);
                repeated.solve();

                auto block       = work.getView();
                auto tiled       = rhoLong.getView();
                const int nghost = rho.getNghost();
                Kokkos::parallel_for(
                    "tile rho", rhoLong.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                        tiled(i, j, k) = block(i, j, (k - nghost) % N + nghost);
                    });
                ippl::ParameterList plain = solverParameters(greens);
                Solver_t explicitTrain(rhoLong, plain);
                explicitTrain.solve();

                auto centre = phiReference.getView();
                Kokkos::parallel_for(
                    "central block", phiReference.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const int i, const int j, const int k) {
                        centre(i, j, k) = tiled(i, j, k + K * N);
                    });
                ok &= report(msg, name + " repeated kernel vs explicit train",
                             relativeDifference(rho, phiReference), 1e-10);
            }
        }
    }
    ippl::finalize();
    return ok ? 0 : 1;
}
