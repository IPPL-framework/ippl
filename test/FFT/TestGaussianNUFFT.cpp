/*!
 * @file TestGaussianNUFFT.cpp
 * @brief Integration and scaling driver for Gaussian non-uniform FFT workloads.
 *
 * This executable exercises the IPPL NUFFT transform on a smooth Gaussian
 * particle/source distribution in a way that is useful both as a lightweight
 * integration test and as a manual scaling driver.
 *
 * The physical domain is the periodic cube
 *
 * \f[
 *     \Omega = [0, 2\pi)^3 .
 * \f]
 *
 * A set of non-uniform particle locations \f$x_j \in \Omega\f$ is generated
 * deterministically from a low-discrepancy sequence.  The particle strengths
 * are sampled from an unnormalised Gaussian centered at
 * \f$c=(\pi,\pi,\pi)\f$,
 *
 * \f[
 *     q_j = \exp\left(-\frac{\|x_j-c\|^2}{2\sigma^2}\right).
 * \f]
 *
 * For Type-1 NUFFT the executable measures the particle-to-mode transform
 *
 * \f[
 *     \hat f_k = \sum_{j=0}^{M-1} q_j \exp(i k \cdot x_j),
 *     \qquad k \in \mathcal{K}_N ,
 * \f]
 *
 * where \f$\mathcal{K}_N\f$ is represented by the uniform complex IPPL field.
 * This path primarily stresses particle spreading, the distributed FFT backend,
 * and post-FFT deconvolution in the native implementation, or the configured
 * FINUFFT/cuFINUFFT backend when requested.
 *
 * For Type-2 NUFFT the executable initializes the uniform mode grid with a
 * sparse, analytically known Fourier series
 *
 * \f[
 *     \hat f_k \neq 0 \quad \hbox{only for a small validation set}
 *     \quad \mathcal{V} \subset \mathcal{K}_N .
 * \f]
 *
 * and measures the mode-to-particle interpolation
 *
 * \f[
 *     q_j = \sum_{k \in \mathcal{K}_N} \hat f_k \exp(-i k \cdot x_j).
 * \f]
 *
 * This path stresses pre-correction, inverse FFT work, and particle gather.
 * The sparse series keeps validation cheap: every particle value can be checked
 * against the closed-form sum over \f$\mathcal{V}\f$ without performing a full
 * direct \f$\mathcal{O}(MN^3)\f$ reference transform.
 *
 * Validation is deliberately stronger than a nonzero norm check.  Type-1
 * compares several low-frequency output modes against direct particle sums,
 * and Type-2 compares particle values against the sparse analytic Fourier
 * series.  Validation is performed outside the timed region.
 *
 * The timing region intentionally excludes mesh/layout construction, particle
 * creation, deterministic initialization, and final sanity checks.  A warmup
 * transform is executed before measured iterations so backend plan setup and
 * first-touch effects are not mixed into the steady-state transform timing.
 *
 * Usage:
 *
 * \code
 *   TestGaussianNUFFT <nx> <ny> <nz> <num_particles> <iterations>
 *                     <type> <backend> [method] --info 5
 * \endcode
 *
 * Arguments:
 * - \c nx, \c ny, \c nz: global mode-grid extents.
 * - \c num_particles: global number of particles before redistribution.
 * - \c iterations: number of measured transform calls.
 * - \c type: \c 1 for particles-to-grid, \c 2 for grid-to-particles.
 * - \c backend: \c native or \c finufft.  In GPU FINUFFT builds, \c finufft
 *   selects the configured cuFINUFFT backend and is valid only for one MPI rank.
 * - \c method: optional native spread/gather method.  For Type-1 this controls
 *   \c spread_method; for Type-2 this controls \c gather_method.
 *
 * Example:
 *
 * \code
 *   srun -n 4 ./TestGaussianNUFFT 64 64 64 1000000 10 1 native tiled --info 5
 * \endcode
 */

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"
#include "Utility/ParameterList.h"

namespace {

    constexpr unsigned Dim = 3;
    constexpr int NumValidationModes = 5;
    constexpr double ValidationTolerance = 1.0e-3;

    using complex_type = Kokkos::complex<double>;

    template <class PLayout>
    struct GaussianBunch : public ippl::ParticleBase<PLayout> {
        using execution_space = typename PLayout::position_execution_space;
        using charge_type     = ippl::ParticleAttrib<double, execution_space>;

        explicit GaussianBunch(PLayout& layout)
            : ippl::ParticleBase<PLayout>(layout) {
            this->addAttribute(Q);
        }

        charge_type Q;
    };

    double halton(std::uint64_t index, std::uint64_t base) {
        double result = 0.0;
        double f      = 1.0 / static_cast<double>(base);

        while (index > 0) {
            result += f * static_cast<double>(index % base);
            index /= base;
            f /= static_cast<double>(base);
        }

        return result;
    }

    std::size_t localParticleCount(std::size_t total) {
        const auto size = static_cast<std::size_t>(ippl::Comm->size());
        const auto rank = static_cast<std::size_t>(ippl::Comm->rank());
        return total / size + ((rank < total % size) ? 1 : 0);
    }

    std::size_t firstGlobalParticle(std::size_t total) {
        const auto size = static_cast<std::size_t>(ippl::Comm->size());
        const auto rank = static_cast<std::size_t>(ippl::Comm->rank());
        const auto base = total / size;
        const auto rem  = total % size;
        return rank * base + std::min(rank, rem);
    }

    template <class Bunch>
    void initializeGaussianParticles(Bunch& bunch, std::size_t totalParticles) {
        const double twoPi = 2.0 * Kokkos::numbers::pi_v<double>;
        const double mu    = Kokkos::numbers::pi_v<double>;
        const double sigma = 0.35 * Kokkos::numbers::pi_v<double>;

        const auto localCount = localParticleCount(totalParticles);
        const auto first      = firstGlobalParticle(totalParticles);

        bunch.create(localCount);

        auto rHost = bunch.R.getHostMirror();
        auto qHost = bunch.Q.getHostMirror();

        for (std::size_t i = 0; i < localCount; ++i) {
            const auto gid = first + i + 1;

            ippl::Vector<double, Dim> r;
            r[0] = twoPi * halton(gid, 2);
            r[1] = twoPi * halton(gid, 3);
            r[2] = twoPi * halton(gid, 5);

            const double dx = r[0] - mu;
            const double dy = r[1] - mu;
            const double dz = r[2] - mu;
            const double r2 = dx * dx + dy * dy + dz * dz;

            rHost(i) = r;
            qHost(i) = std::exp(-r2 / (2.0 * sigma * sigma));
        }

        Kokkos::deep_copy(bunch.R.getView(), rHost);
        Kokkos::deep_copy(bunch.Q.getView(), qHost);
        Kokkos::fence();

        bunch.update();
    }

    template <class Field>
    void initializeSparseValidationSpectrum(Field& field) {
        auto view          = field.getView();
        const auto& layout = field.getLayout();
        const auto& ldom   = layout.getLocalNDIndex();
        const int nghost   = field.getNghost();

        const int nx = layout.getDomain()[0].length();
        const int ny = layout.getDomain()[1].length();
        const int nz = layout.getDomain()[2].length();

        Kokkos::Array<int, NumValidationModes> kx = {0, 1, -1, 0, 0};
        Kokkos::Array<int, NumValidationModes> ky = {0, 0, 0, 1, -1};
        Kokkos::Array<int, NumValidationModes> kz = {0, 0, 0, 1, -1};
        Kokkos::Array<double, NumValidationModes> cr = {0.75, 0.20, 0.20, 0.11, 0.11};
        Kokkos::Array<double, NumValidationModes> ci = {0.00, 0.10, -0.10, -0.04, 0.04};

        Kokkos::parallel_for(
            "InitializeGaussianNUFFTSparseSpectrum", field.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                complex_type value(0.0, 0.0);
                for (int m = 0; m < NumValidationModes; ++m) {
                    const int ix = (kx[m] >= 0) ? kx[m] : nx + kx[m];
                    const int iy = (ky[m] >= 0) ? ky[m] : ny + ky[m];
                    const int iz = (kz[m] >= 0) ? kz[m] : nz + kz[m];
                    if (ig == ix && jg == iy && kg == iz) {
                        value = complex_type(cr[m], ci[m]);
                    }
                }
                view(i, j, k) = value;
            });

        Kokkos::fence();
        field.fillHalo();
    }

    template <class Bunch>
    complex_type computeType1ReferenceMode(const Bunch& bunch, const ippl::Vector<int, Dim>& mode) {
        using exec_space = typename Bunch::execution_space;

        auto rView = bunch.R.getView();
        auto qView = bunch.Q.getView();
        const auto nloc = bunch.getLocalNum();
        const complex_type imag(0.0, 1.0);
        const auto k = mode;

        complex_type local(0.0, 0.0);
        Kokkos::parallel_reduce(
            "GaussianNUFFTValidateType1Mode", Kokkos::RangePolicy<exec_space>(0, nloc),
            KOKKOS_LAMBDA(const std::size_t i, complex_type& value) {
                double phase = 0.0;
                for (unsigned d = 0; d < Dim; ++d) {
                    phase += static_cast<double>(k[d]) * rView(i)[d];
                }
                value += (Kokkos::cos(phase) - imag * Kokkos::sin(phase)) * qView(i);
            },
            Kokkos::Sum<complex_type>(local));

        double send[2] = {local.real(), local.imag()};
        double recv[2] = {0.0, 0.0};
        MPI_Allreduce(send, recv, 2, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
        return complex_type(recv[0], recv[1]);
    }

    template <class Field>
    complex_type readMode(const Field& field, const ippl::Vector<int, Dim>& mode) {
        const auto& layout = field.getLayout();
        const auto& ldom   = layout.getLocalNDIndex();
        const auto& domain = layout.getDomain();
        const int nghost   = field.getNghost();

        ippl::Vector<int, Dim> global;
        for (unsigned d = 0; d < Dim; ++d) {
            const int n = domain[d].length();
            global[d]   = (mode[d] >= 0) ? mode[d] : n + mode[d];
        }

        complex_type local(0.0, 0.0);
        bool owned = true;
        for (unsigned d = 0; d < Dim; ++d) {
            owned = owned && global[d] >= ldom[d].first() && global[d] <= ldom[d].last();
        }

        if (owned) {
            auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.getView());
            local = host(global[0] - ldom[0].first() + nghost,
                         global[1] - ldom[1].first() + nghost,
                         global[2] - ldom[2].first() + nghost);
        }

        double send[2] = {local.real(), local.imag()};
        double recv[2] = {0.0, 0.0};
        MPI_Allreduce(send, recv, 2, MPI_DOUBLE, MPI_SUM, ippl::Comm->getCommunicator());
        return complex_type(recv[0], recv[1]);
    }

    template <class Bunch, class Field>
    double validateType1(const Bunch& bunch, const Field& field) {
        const std::array<ippl::Vector<int, Dim>, NumValidationModes> modes = {
            ippl::Vector<int, Dim>{0, 0, 0}, ippl::Vector<int, Dim>{1, 0, 0},
            ippl::Vector<int, Dim>{0, 1, 0}, ippl::Vector<int, Dim>{0, 0, 1},
            ippl::Vector<int, Dim>{1, -1, 1}};

        double maxRelativeError = 0.0;
        for (const auto& mode : modes) {
            const auto reference = computeType1ReferenceMode(bunch, mode);
            const auto actual    = readMode(field, mode);
            const double error   = Kokkos::abs(actual - reference);
            const double scale   = std::max(1.0, static_cast<double>(Kokkos::abs(reference)));
            maxRelativeError     = std::max(maxRelativeError, error / scale);
        }

        return maxRelativeError;
    }

    template <class Bunch>
    double validateType2(const Bunch& bunch) {
        using exec_space = typename Bunch::execution_space;

        auto rView = bunch.R.getView();
        auto qView = bunch.Q.getView();
        const auto nloc = bunch.getLocalNum();
        const complex_type imag(0.0, 1.0);

        Kokkos::Array<int, NumValidationModes> kx = {0, 1, -1, 0, 0};
        Kokkos::Array<int, NumValidationModes> ky = {0, 0, 0, 1, -1};
        Kokkos::Array<int, NumValidationModes> kz = {0, 0, 0, 1, -1};
        Kokkos::Array<double, NumValidationModes> cr = {0.75, 0.20, 0.20, 0.11, 0.11};
        Kokkos::Array<double, NumValidationModes> ci = {0.00, 0.10, -0.10, -0.04, 0.04};

        double localMax = 0.0;
        Kokkos::parallel_reduce(
            "GaussianNUFFTValidateType2", Kokkos::RangePolicy<exec_space>(0, nloc),
            KOKKOS_LAMBDA(const std::size_t i, double& maxError) {
                complex_type expected(0.0, 0.0);
                for (int m = 0; m < NumValidationModes; ++m) {
                    const double phase = kx[m] * rView(i)[0] + ky[m] * rView(i)[1]
                                         + kz[m] * rView(i)[2];
                    const complex_type coeff(cr[m], ci[m]);
                    expected += coeff * (Kokkos::cos(phase) + imag * Kokkos::sin(phase));
                }

                const double scale = Kokkos::max(1.0, Kokkos::abs(expected.real()));
                const double error = Kokkos::abs(qView(i) - expected.real()) / scale;
                if (error > maxError) {
                    maxError = error;
                }
            },
            Kokkos::Max<double>(localMax));

        double globalMax = 0.0;
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX,
                      ippl::Comm->getCommunicator());
        return globalMax;
    }

    ippl::ParameterList makeNUFFTParams(const std::string& backend, const std::string& method,
                                        int type) {
        ippl::ParameterList params;
        params.add("tolerance", 1.0e-6);
        params.add("use_upsampled_inputs", false);
        params.add("use_kokkos_nufft", false);
        params.add("sort", true);
        params.add("tile_size_3d", 6);
        params.add("z_tiles", 1);

        if (backend == "native") {
            params.add("use_finufft", false);
        } else if (backend == "finufft") {
#ifndef ENABLE_FINUFFT
            throw IpplException("TestGaussianNUFFT", "FINUFFT backend requested but not enabled");
#else
#ifdef ENABLE_GPU_NUFFT
            if (ippl::Comm->size() != 1) {
                throw IpplException("TestGaussianNUFFT",
                                    "cuFINUFFT backend has no MPI decomposition; use one rank");
            }
#endif
            params.add("use_finufft", true);
            params.add("use_finufft_defaults", false);
#endif
        } else {
            throw IpplException("TestGaussianNUFFT", "Unknown NUFFT backend");
        }

        if (type == 1) {
            params.add("spread_method", method);
            params.add("gather_method", std::string("atomic_sort"));
        } else if (type == 2) {
            params.add("spread_method", std::string("tiled"));
            params.add("gather_method", method);
        } else {
            throw IpplException("TestGaussianNUFFT", "NUFFT type must be 1 or 2");
        }

#ifdef ENABLE_GPU_NUFFT
        params.add("gpu_method", 1);
        params.add("gpu_sort", 0);
        params.add("gpu_kerevalmeth", 1);
#else
        params.add("spread_kerevalmeth", 1);
        params.add("spread_sort", 2);
        params.add("nthreads", 0);
#endif

        return params;
    }

    void printUsage(const char* executable) {
        if (ippl::Comm->rank() == 0) {
            std::cerr << "Usage: " << executable
                      << " <nx> <ny> <nz> <num_particles> <iterations>"
                      << " <type> <backend> [method] --info 5\n";
        }
    }

}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);

        if (argc < 8) {
            printUsage(argv[0]);
            ippl::finalize();
            return 1;
        }

        using mesh_type          = ippl::UniformCartesian<double, Dim>;
        using centering_type     = typename mesh_type::DefaultCentering;
        using real_field_type    = ippl::Field<double, Dim, mesh_type, centering_type>;
        using execution_space    = typename real_field_type::execution_space;
        using fft_type           = ippl::FFT<ippl::NUFFTransform, real_field_type>;
        using complex_field_type = typename fft_type::ComplexField;
        using playout_type       = ippl::ParticleSpatialLayout<double, Dim, mesh_type, execution_space>;
        using bunch_type         = GaussianBunch<playout_type>;

        const int nx = std::atoi(argv[1]);
        const int ny = std::atoi(argv[2]);
        const int nz = std::atoi(argv[3]);
        const std::size_t totalParticles = static_cast<std::size_t>(std::atoll(argv[4]));
        const int iterations             = std::atoi(argv[5]);
        const int type                   = std::atoi(argv[6]);
        const std::string backend        = argv[7];
        const std::string method         = (argc > 8 && argv[8][0] != '-') ? argv[8] : "tiled";

        if (nx < 4 || ny < 4 || nz < 4 || iterations <= 0 || totalParticles == 0) {
            throw IpplException("TestGaussianNUFFT",
                                "Grid extents must be at least four; particle count and iterations "
                                "must be positive");
        }

        msg << "TestGaussianNUFFT: grid = (" << nx << ", " << ny << ", " << nz
            << "), particles = " << totalParticles << ", iterations = " << iterations
            << ", type = " << type << ", backend = " << backend << ", method = " << method
            << endl;

        static IpplTimings::TimerRef totalTimer = IpplTimings::getTimer("GaussianNUFFT::total");
        IpplTimings::startTimer(totalTimer);

        ippl::NDIndex<Dim> owned{ippl::Index(nx), ippl::Index(ny), ippl::Index(nz)};

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel);

        const double twoPi = 2.0 * Kokkos::numbers::pi_v<double>;
        ippl::Vector<double, Dim> hx     = {twoPi / nx, twoPi / ny, twoPi / nz};
        ippl::Vector<double, Dim> origin = {0.0, 0.0, 0.0};
        mesh_type mesh(owned, hx, origin);

        real_field_type realField(mesh, layout);
        complex_field_type modes(mesh, layout);

        playout_type playout(layout, mesh);
        bunch_type bunch(playout);
        bunch.setParticleBC(ippl::BC::PERIODIC);

        static IpplTimings::TimerRef initTimer = IpplTimings::getTimer("GaussianNUFFT::initialize");
        IpplTimings::startTimer(initTimer);
        initializeGaussianParticles(bunch, totalParticles);
        initializeSparseValidationSpectrum(modes);
        auto params = makeNUFFTParams(backend, method, type);
        fft_type fft(layout, bunch.getLocalNum(), type, params);
        IpplTimings::stopTimer(initTimer);

        static IpplTimings::TimerRef warmupTimer = IpplTimings::getTimer("GaussianNUFFT::warmup");
        IpplTimings::startTimer(warmupTimer);
        if (type == 1) {
            Kokkos::deep_copy(modes.getView(), typename complex_field_type::value_type(0.0, 0.0));
        } else {
            initializeSparseValidationSpectrum(modes);
            Kokkos::deep_copy(bunch.Q.getView(), 0.0);
        }
        fft.transform(bunch.R, bunch.Q, modes);
        Kokkos::fence();
        IpplTimings::stopTimer(warmupTimer);

        static IpplTimings::TimerRef transformTimer =
            IpplTimings::getTimer("GaussianNUFFT::transform");
        for (int iter = 0; iter < iterations; ++iter) {
            if (type == 1) {
                Kokkos::deep_copy(modes.getView(), typename complex_field_type::value_type(0.0, 0.0));
            } else {
                initializeSparseValidationSpectrum(modes);
                Kokkos::deep_copy(bunch.Q.getView(), 0.0);
            }

            IpplTimings::startTimer(transformTimer);
            fft.transform(bunch.R, bunch.Q, modes);
            Kokkos::fence();
            IpplTimings::stopTimer(transformTimer);
        }

        const double validationError =
            (type == 1) ? validateType1(bunch, modes) : validateType2(bunch);
        if (!std::isfinite(validationError) || validationError > ValidationTolerance) {
            std::ostringstream os;
            os << "Validation failed: relative error = " << validationError
               << ", tolerance = " << ValidationTolerance;
            throw IpplException("TestGaussianNUFFT", os.str());
        }

        msg << "GaussianNUFFT validation relative error = " << std::setprecision(16)
            << validationError << endl;
        IpplTimings::stopTimer(totalTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
