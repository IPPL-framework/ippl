/**
 * @file CosmologyICCheck.hpp
 * @brief Collective acceptance checks for initialized cosmological phase space.
 * @ingroup cosmology_core
 *
 * The native gate checks the Lagrangian 1LPT state before the first force solve.
 * It reads existing device particle attributes and initial Fourier coefficients;
 * it neither changes their values nor gathers a particle catalogue on the host.
 * Additional storage consists of radial tables and shell histograms, not meshes.
 */
#ifndef IPPL_COSMOLOGY_IC_CHECK_HPP
#define IPPL_COSMOLOGY_IC_CHECK_HPP

#include <array>
#include <limits>
#include <sstream>
#include <utility>

namespace cosmology {
namespace iccheck_detail {

/** @brief Write finite diagnostic numbers as JSON numbers and invalid values as null.
 * @param out Host report stream.
 * @param value Diagnostic value; nonfinite values remain evidence of a failed gate.
 */
inline void writeNumber(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value;
    else out << "null";
}

/** @brief One independent complex check of a native initial displacement mode. */
struct ModeCheck {
    std::array<int, 3> mode_m{}; ///< Signed physical Fourier mode, independent of grid ordering.
    double expectedReal_m = 0; ///< Real part of the initial density coefficient D(a) delta_0.
    double expectedImaginary_m = 0; ///< Imaginary part of the initial density coefficient D(a) delta_0.
    double recoveredReal_m = 0; ///< Real part of the coefficient from the displacement DFT.
    double recoveredImaginary_m = 0; ///< Imaginary part of the coefficient from the displacement DFT.
    double error_m = 0; ///< Absolute complex error in the displacement recovery.
    double tolerance_m = 0; ///< Roundoff budget for the displacement recovery.
    bool passed_m = false; ///< True when the absolute complex error meets its budget.
    double declaredReal_m = 0; ///< Real part of the initial physical coefficient from the declared seed/configuration.
    double declaredImaginary_m = 0; ///< Imaginary part of the initial physical coefficient from the declared seed/configuration.
    double declaredError_m = 0; ///< Absolute stored-versus-declared coefficient error.
    double declaredTolerance_m = 0; ///< Roundoff budget for the stored-versus-declared coefficient comparison.
    bool declaredPassed_m = false; ///< True when the retained coefficient matches the declared realization.
};

/** @brief Shell average of the native linear input field, not a particle-density estimator. */
struct Shell {
    int index_m = 0; ///< Integer shell floor(|m|), with width one fundamental wave number.
    std::uint64_t modes_m = 0; ///< Both members of each Hermitian pair are counted.
    double waveNumber_m = 0; ///< Mean wave number k in this shell.
    double power_m = 0; ///< Mean measured D^2 V |delta_0|^2 in this shell.
    double target_m = 0; ///< Mean target D^2 P(k) in this shell.
};
} // namespace iccheck_detail

/**
 * @brief Accept or reject the initialized state before any force solve or evolution.
 *
 * Native generation obeys
 * \f$x(q)=q+D(a)\Psi_0(q)\f$ and
 * \f$p(q)=a^2E(a)f(a)[x(q)-q]_{\rm periodic}\f$.
 * For each selected physical mode, the gate independently reduces
 * \f$\widehat\delta_{\rm recovered}(k)=-i k\cdot
 * N^{-1}\sum_q[x(q)-q]_{\rm periodic}e^{-ik\cdot q}\f$
 * and compares it with the stored initial coefficient after removing the
 * half-cell phase \f$\exp[i\pi(m_x+m_y+m_z)/n]\f$ and multiplying by D(a).
 *
 * Per-particle momentum budgets include coordinate-subtraction error
 * proportional to epsilon_double * L and, when requested, float32 rounding.
 * Fourier budgets include coordinate cancellation and reduction/trigonometric
 * roundoff. The minimal-image check requires native initial displacements to
 * be smaller than L/2 in each component; this condition is checked explicitly.
 *
 * Exact counts, unit PM weights, finite wrapped phase space, ID range and two
 * expected modulo-2^64 ID signatures are checked for all input modes. The
 * signatures detect corruption but are not a mathematical proof of uniqueness.
 * External phase space is not assumed to satisfy a native 1LPT relation or to
 * use the same gravitational constant in its physical mass convention.
 *
 * pk_initial.csv contains shell averages of the native LINEAR FIELD:
 * \f$P_{\rm initial}=D(a)^2 V|\delta_0(k)|^2\f$. It is not the spectrum of CIC
 * deposited particles, and has no shot-noise subtraction. Gaussian realization
 * scatter around D(a)^2 P(k) is reported as a warning, never a rejection.
 *
 * All MPI ranks enter collectively. Device arrays are read only; shell tables
 * and scalar reductions are copied to host memory for rank-zero reports.
 * @pre Native modes_m still contains initial z=0 density coefficients, q stores
 * the undisplaced lattice, and wrapping/migration and particleMass_m assignment
 * have completed. The simulation output directory exists.
 * @throws std::runtime_error After publishing a failed ic_check.json, or if the
 * collective report cannot be written.
 */
inline void Simulation::validateInitialConditions() {
    const int rank = ippl::Comm->rank(), ranks = ippl::Comm->size();
    const auto communicator = ippl::Comm->getCommunicator();
    const bool native = config_m.icMode != "external";
    const bool gaussian = config_m.icMode == "gaussian";
    const bool sine = config_m.icMode == "sine";
    const int n = config_m.nGrid;
    const double box = config_m.boxSize, cell = box / n;
    const double epsilon = std::numeric_limits<double>::epsilon();
    const double a = config_m.aInitial(), growth = background_m.D(a);
    const double momentumFactor = a * a * background_m.E(a) * background_m.f(a);
    const double fundamental = 2 * Kokkos::numbers::pi_v<double> / box;
    const double volume = box * box * box;
    const auto expectedCount = totalParticles();
    const auto localCount = particles_m->getLocalNum();
    const auto positions = particles_m->R.getView();
    const auto momenta = particles_m->momentum.getView();
    const auto lagrangian = particles_m->lagrangian.getView();
    const auto weights = particles_m->mass.getView();
    const auto ids = particles_m->globalId.getView();
    using ParticlePolicy = Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>;

    std::uint64_t observedCount = 0, localCount64 = localCount;
    MPI_Allreduce(&localCount64, &observedCount, 1, MPI_UINT64_T, MPI_SUM, communicator);
    std::uint64_t badPhase = 0, badWeights = 0, badIds = 0, idSum = 0, idHash = 0, badLattice = 0;
    Kokkos::parallel_reduce("Cosmology IC particle invariants", ParticlePolicy(0, localCount),
        KOKKOS_LAMBDA(const std::size_t i, std::uint64_t& phase, std::uint64_t& mass,
                      std::uint64_t& badId, std::uint64_t& sum, std::uint64_t& hash,
                      std::uint64_t& lattice) {
            const auto id = ids(i);
            bool invalid = false, invalidLattice = false;
            for (int d = 0; d < 3; ++d) {
                invalid = invalid || !Kokkos::isfinite(positions(i)[d])
                    || !Kokkos::isfinite(momenta(i)[d]) || !Kokkos::isfinite(lagrangian(i)[d])
                    || positions(i)[d] < 0 || positions(i)[d] >= box;
            }
            if (native && id < expectedCount) {
                const auto grid = static_cast<std::uint64_t>(n);
                const std::uint64_t coordinate[3] = {id % grid, (id / grid) % grid, id / (grid * grid)};
                for (int d = 0; d < 3; ++d) {
                    const double expected = (static_cast<double>(coordinate[d]) + 0.5) * cell;
                    invalidLattice = invalidLattice || !Kokkos::isfinite(lagrangian(i)[d])
                        || Kokkos::abs(lagrangian(i)[d] - expected) > 8 * epsilon * box;
                }
            }
            phase += invalid;
            mass += (!Kokkos::isfinite(weights(i)) || weights(i) != 1.0);
            badId += (id >= expectedCount);
            lattice += invalidLattice;
            sum += id;
            hash += mixBits(id);
        }, badPhase, badWeights, badIds, idSum, idHash, badLattice);
    const std::uint64_t localInvariants[6] = {badPhase, badWeights, badIds, idSum, idHash, badLattice};
    std::uint64_t invariants[6];
    MPI_Allreduce(localInvariants, invariants, 6, MPI_UINT64_T, MPI_SUM, communicator);

    // Generate the expected signature without retaining or gathering any labels.
    const std::uint64_t referenceCount = expectedCount / ranks + (std::uint64_t(rank) < expectedCount % ranks);
    const std::uint64_t referenceFirst = expectedCount / ranks * rank
        + std::min(std::uint64_t(rank), expectedCount % ranks);
    std::uint64_t localReferenceHash = 0, referenceHash = 0;
    Kokkos::parallel_reduce("Cosmology IC expected labels",
        ParticlePolicy(0, static_cast<std::size_t>(referenceCount)),
        KOKKOS_LAMBDA(const std::size_t i, std::uint64_t& hash) {
            hash += mixBits(referenceFirst + i);
        }, localReferenceHash);
    MPI_Allreduce(&localReferenceHash, &referenceHash, 1, MPI_UINT64_T, MPI_SUM, communicator);
    // Divide an even factor before multiplication; the remaining uint64 overflow
    // intentionally implements the same modulo-2^64 signature as the reduction.
    const std::uint64_t referenceSum = expectedCount % 2 == 0
        ? (expectedCount / 2) * (expectedCount - 1)
        : expectedCount * ((expectedCount - 1) / 2);
    const bool countPassed = observedCount == expectedCount;
    const bool phasePassed = invariants[0] == 0;
    const bool weightsPassed = invariants[1] == 0;
    const bool idsPassed = invariants[2] == 0 && invariants[3] == referenceSum
        && invariants[4] == referenceHash;
    const bool latticePassed = !native || invariants[5] == 0;
    const double impliedMass = 2.77536627e11 * config_m.omegaMatter * volume / double(expectedCount);
    const double physicalMassRatio = particleMass_m / impliedMass;
    const bool physicalMassPassed = std::isfinite(particleMass_m) && particleMass_m > 0
        && (!native || std::abs(physicalMassRatio - 1) <= 32 * epsilon);

    double momentumErrorRatio = 0, maximumDisplacement = 0, maximumMomentumDisplacement = 0;
    const double momentumEpsilon = config_m.icMomentumPrecision == "float32"
        ? double(std::numeric_limits<float>::epsilon()) : epsilon;
    if (native && phasePassed) {
        Kokkos::parallel_reduce("Cosmology IC momentum relation", ParticlePolicy(0, localCount),
            KOKKOS_LAMBDA(const std::size_t i, double& ratio, double& displacementMaximum,
                          double& momentumDisplacementMaximum) {
                for (int d = 0; d < 3; ++d) {
                    double displacement = positions(i)[d] - lagrangian(i)[d];
                    displacement -= box * Kokkos::floor(displacement / box + 0.5);
                    const double expected = momentumFactor * displacement;
                    const double tolerance = 32 * epsilon * box * momentumFactor
                        + 4 * momentumEpsilon * Kokkos::max(Kokkos::abs(expected), Kokkos::abs(momenta(i)[d]));
                    ratio = Kokkos::max(ratio, Kokkos::abs(momenta(i)[d] - expected) / tolerance);
                    displacementMaximum = Kokkos::max(displacementMaximum, Kokkos::abs(displacement));
                    momentumDisplacementMaximum = Kokkos::max(momentumDisplacementMaximum,
                        Kokkos::abs(momenta(i)[d]) / momentumFactor);
                }
            }, Kokkos::Max<double>(momentumErrorRatio), Kokkos::Max<double>(maximumDisplacement),
               Kokkos::Max<double>(maximumMomentumDisplacement));
        momentumErrorRatio = globalMax(momentumErrorRatio);
        maximumDisplacement = globalMax(maximumDisplacement);
        maximumMomentumDisplacement = globalMax(maximumMomentumDisplacement);
    }
    const bool momentumPassed = !native || (phasePassed && std::isfinite(momentumErrorRatio)
        && momentumErrorRatio <= 1 && maximumMomentumDisplacement < box / 2);

    bool finiteModesPassed = true, dcPassed = true, cutoffPassed = true, realityPassed = true;
    bool declaredModesPassed = true;
    double realityTolerance = 0, displacementBound = 0;
    std::vector<iccheck_detail::Shell> shells;
    std::vector<iccheck_detail::ModeCheck> modeChecks;
    int spectrumWarnings = 0;
    if (native) {
        const int cutoff = config_m.icModeCutoff;
        const int maximumSquaredMode = cutoff > 0 ? cutoff * cutoff : 3 * (n / 2) * (n / 2);
        const int bins = int(std::sqrt(double(maximumSquaredMode))) + 1;
        Kokkos::View<double*> radialPower("Cosmology IC target spectrum", maximumSquaredMode + 1);
        auto radialHost = Kokkos::create_mirror_view(radialPower);
        std::unique_ptr<PowerSpectrum> spectrum;
        if (gaussian) {
            spectrum = std::make_unique<PowerSpectrum>(config_m);
            radialHost(0) = 0;
            for (int squared = 1; squared <= maximumSquaredMode; ++squared)
                radialHost(squared) = (*spectrum)(fundamental * std::sqrt(double(squared)));
        } else {
            for (int squared = 0; squared <= maximumSquaredMode; ++squared) radialHost(squared) = 0;
        }
        Kokkos::deep_copy(radialPower, radialHost);
        Kokkos::View<double**> histogram("Cosmology IC spectrum sums", bins, 3);
        Kokkos::View<std::uint64_t*> modeCounts("Cosmology IC mode counts", bins);
        Kokkos::View<std::uint64_t*> invalidModes("Cosmology IC invalid modes", 3);
        Kokkos::deep_copy(histogram, 0.0);
        Kokkos::deep_copy(modeCounts, std::uint64_t(0));
        Kokkos::deep_copy(invalidModes, std::uint64_t(0));
        const auto view = modes_m.getView();
        const auto domain = layout_m->getLocalNDIndex();
        const int ghosts = modes_m.getNghost();
        const ippl::Vector<int, 3> sineMode(config_m.mode[0], config_m.mode[1], config_m.mode[2]);
        const double sinePower = volume * config_m.amplitude * config_m.amplitude / 4;
        ippl::parallel_reduce("Cosmology IC linear spectrum", ippl::getRangePolicy(view, ghosts),
            KOKKOS_LAMBDA(const Index& index, double& bound) {
                int mode[3], squared = 0;
                bool nyquist = false, positiveSine = true, negativeSine = true;
                for (int d = 0; d < 3; ++d) {
                    const int global = index[d] - ghosts + domain[d].first();
                    mode[d] = global <= n / 2 ? global : global - n;
                    squared += mode[d] * mode[d];
                    nyquist = nyquist || global == n / 2;
                    positiveSine = positiveSine && mode[d] == sineMode[d];
                    negativeSine = negativeSine && mode[d] == -sineMode[d];
                }
                const auto coefficient = ippl::apply(view, index);
                if (!Kokkos::isfinite(coefficient.real()) || !Kokkos::isfinite(coefficient.imag())) {
                    Kokkos::atomic_add(&invalidModes(0), std::uint64_t(1));
                    return;
                }
                const double normSquared = coefficient.real() * coefficient.real()
                    + coefficient.imag() * coefficient.imag();
                if (!Kokkos::isfinite(normSquared)) {
                    Kokkos::atomic_add(&invalidModes(0), std::uint64_t(1));
                    return;
                }
                if (squared == 0) {
                    if (coefficient.real() != 0 || coefficient.imag() != 0)
                        Kokkos::atomic_add(&invalidModes(1), std::uint64_t(1));
                    return;
                }
                if (nyquist || squared > maximumSquaredMode) {
                    if (coefficient.real() != 0 || coefficient.imag() != 0)
                        Kokkos::atomic_add(&invalidModes(2), std::uint64_t(1));
                    return;
                }
                const double radius = Kokkos::sqrt(double(squared));
                const int shell = int(radius);
                const double measured = growth * growth * volume * normSquared;
                const double target = gaussian ? growth * growth * radialPower(squared)
                    : ((sine && (positiveSine || negativeSine)) ? sinePower : 0);
                if (!Kokkos::isfinite(measured) || !Kokkos::isfinite(target)) {
                    Kokkos::atomic_add(&invalidModes(0), std::uint64_t(1));
                    return;
                }
                Kokkos::atomic_add(&modeCounts(shell), std::uint64_t(1));
                Kokkos::atomic_add(&histogram(shell, 0), fundamental * radius);
                Kokkos::atomic_add(&histogram(shell, 1), measured);
                Kokkos::atomic_add(&histogram(shell, 2), target);
                bound += Kokkos::sqrt(normSquared) / (fundamental * radius);
            }, displacementBound);
        displacementBound = globalSum(displacementBound);
        auto histogramHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), histogram);
        auto countHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), modeCounts);
        auto invalidHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), invalidModes);
        std::vector<double> localSums(3 * bins), globalSums(3 * bins);
        std::vector<std::uint64_t> localCounts(bins), globalCounts(bins);
        for (int bin = 0; bin < bins; ++bin) {
            localCounts[bin] = countHost(bin);
            for (int d = 0; d < 3; ++d) localSums[3 * bin + d] = histogramHost(bin, d);
        }
        MPI_Allreduce(localSums.data(), globalSums.data(), 3 * bins, MPI_DOUBLE, MPI_SUM, communicator);
        MPI_Allreduce(localCounts.data(), globalCounts.data(), bins, MPI_UINT64_T, MPI_SUM, communicator);
        const std::uint64_t localInvalid[3] = {invalidHost(0), invalidHost(1), invalidHost(2)};
        std::uint64_t invalid[3];
        MPI_Allreduce(localInvalid, invalid, 3, MPI_UINT64_T, MPI_SUM, communicator);
        finiteModesPassed = invalid[0] == 0 && std::isfinite(displacementBound);
        dcPassed = invalid[1] == 0;
        cutoffPassed = invalid[2] == 0;
        realityTolerance = 256 * epsilon * (1 + std::log2(double(n))) * displacementBound;
        realityPassed = std::isfinite(maxImaginary_m) && maxImaginary_m <= realityTolerance;
        for (int bin = 1; bin < bins; ++bin) {
            if (globalCounts[bin] == 0) continue;
            const double count = double(globalCounts[bin]);
            iccheck_detail::Shell shell{bin, globalCounts[bin], globalSums[3 * bin] / count,
                globalSums[3 * bin + 1] / count, globalSums[3 * bin + 2] / count};
            if (gaussian && shell.target_m > 0
                && std::abs(shell.power_m / shell.target_m - 1) > 5 * std::sqrt(2 / count))
                ++spectrumWarnings;
            shells.push_back(shell);
        }

        // The DFT reads particle displacements independently of FFT output order.
        // A bounded selection keeps the cost proportional to Nparticles.
        std::vector<std::array<int, 3>> selected;
        const auto addMode = [&](std::array<int, 3> mode) {
            bool admissible = mode != std::array<int, 3>{0, 0, 0};
            for (const auto component : mode)
                admissible = admissible && component > -n / 2 && component < n / 2;
            if (admissible && std::find(selected.begin(), selected.end(), mode) == selected.end())
                selected.push_back(mode);
        };
        if (sine) addMode(config_m.mode);
        for (const auto mode : {std::array<int, 3>{1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                                {1, 1, 1}, {1, -1, 0}}) addMode(mode);
        const int middle = std::max(1, (cutoff > 0 ? cutoff : n / 2 - 1) / 2);
        addMode({middle, 1, 0});
        if (cutoff > 0) addMode({cutoff + 1, 0, 0});
        if (phasePassed && idsPassed && latticePassed && finiteModesPassed) {
            for (const auto& mode : selected) {
                Complex localCoefficient(0, 0);
                std::array<int, 3> index{};
                bool owns = true;
                for (int d = 0; d < 3; ++d) {
                    const int global = (mode[d] + n) % n;
                    owns = owns && global >= domain[d].first() && global <= domain[d].last();
                    index[d] = global - domain[d].first() + ghosts;
                }
                if (owns)
                    Kokkos::deep_copy(localCoefficient, Kokkos::subview(view, index[0], index[1], index[2]));
                const double coefficientLocal[2] = {localCoefficient.real(), localCoefficient.imag()};
                double coefficientGlobal[2];
                MPI_Allreduce(coefficientLocal, coefficientGlobal, 2, MPI_DOUBLE, MPI_SUM, communicator);
                const double halfCellPhase = Kokkos::numbers::pi_v<double> * (mode[0] + mode[1] + mode[2]) / n;
                const Complex physicalCoefficient = Complex(coefficientGlobal[0], coefficientGlobal[1])
                    * Complex(std::cos(halfCellPhase), -std::sin(halfCellPhase)) * growth;
                // This second expectation is independent of retained modes_m.
                // It catches a coherent but wrong realization/seed, which the
                // displacement-to-retained-mode comparison alone cannot detect.
                const int squared = mode[0] * mode[0] + mode[1] * mode[1] + mode[2] * mode[2];
                Complex declaredCoefficient(0, 0);
                if (squared <= maximumSquaredMode) {
                    if (gaussian) {
                        const double power = (*spectrum)(fundamental * std::sqrt(double(squared)));
                        if (config_m.icRng == "mode_hash_v1") {
                            declaredCoefficient = modeGaussian(config_m.seed, mode[0], mode[1], mode[2])
                                * std::sqrt(power / volume);
                        } else {
                            std::uint64_t global[3], negative[3];
                            for (int d = 0; d < 3; ++d) {
                                global[d] = (mode[d] + n) % n;
                                negative[d] = (n - global[d]) % n;
                            }
                            const auto key = (global[0] * n + global[1]) * n + global[2];
                            const auto other = (negative[0] * n + negative[1]) * n + negative[2];
                            const auto canonical = std::min(key, other);
                            const double u1 = uniformOpen(mixBits(config_m.seed) ^ (2 * canonical));
                            const double u2 = uniformOpen(mixBits(config_m.seed) ^ (2 * canonical + 1));
                            const double radius = std::sqrt(-std::log(u1) * power / volume);
                            const double angle = 2 * Kokkos::numbers::pi_v<double> * u2;
                            declaredCoefficient = Complex(radius * std::cos(angle),
                                radius * std::sin(angle) * (key <= other ? 1 : -1));
                        }
                    } else if (sine && (mode == config_m.mode
                        || mode == std::array<int, 3>{-config_m.mode[0], -config_m.mode[1], -config_m.mode[2]})) {
                        declaredCoefficient = Complex(config_m.amplitude / (2 * growth), 0);
                    }
                }
                declaredCoefficient *= growth;
                const double declaredError = std::hypot(physicalCoefficient.real() - declaredCoefficient.real(),
                    physicalCoefficient.imag() - declaredCoefficient.imag());
                const double declaredTolerance = 64 * epsilon * (std::hypot(declaredCoefficient.real(), declaredCoefficient.imag())
                    + std::hypot(physicalCoefficient.real(), physicalCoefficient.imag()));
                const bool declaredPassed = std::isfinite(declaredError) && declaredError <= declaredTolerance;
                declaredModesPassed = declaredModesPassed && declaredPassed;
                const Vector wave(fundamental * mode[0], fundamental * mode[1], fundamental * mode[2]);
                double real = 0, imaginary = 0;
                Kokkos::parallel_reduce("Cosmology IC recover displacement mode", ParticlePolicy(0, localCount),
                    KOKKOS_LAMBDA(const std::size_t i, double& re, double& im) {
                        double phase = 0, projection = 0;
                        for (int d = 0; d < 3; ++d) {
                            double displacement = positions(i)[d] - lagrangian(i)[d];
                            displacement -= box * Kokkos::floor(displacement / box + 0.5);
                            projection += wave[d] * displacement;
                            phase += wave[d] * lagrangian(i)[d];
                        }
                        re -= projection * Kokkos::sin(phase);
                        im -= projection * Kokkos::cos(phase);
                    }, real, imaginary);
                real = globalSum(real) / double(expectedCount);
                imaginary = globalSum(imaginary) / double(expectedCount);
                const double waveNorm = fundamental * std::sqrt(double(
                    mode[0] * mode[0] + mode[1] * mode[1] + mode[2] * mode[2]));
                const double phaseBound = 2 * Kokkos::numbers::pi_v<double>
                    * (std::abs(mode[0]) + std::abs(mode[1]) + std::abs(mode[2]));
                const double expectedNorm = std::hypot(physicalCoefficient.real(), physicalCoefficient.imag());
                const double tolerance = 128 * epsilon * (1 + std::log2(double(expectedCount)) + phaseBound)
                    * (waveNorm * maximumDisplacement + expectedNorm) + 32 * epsilon * box * waveNorm;
                const double error = std::hypot(real - physicalCoefficient.real(), imaginary - physicalCoefficient.imag());
                modeChecks.push_back({mode, physicalCoefficient.real(), physicalCoefficient.imag(),
                    real, imaginary, error, tolerance, std::isfinite(error) && error <= tolerance,
                    declaredCoefficient.real(), declaredCoefficient.imag(), declaredError, declaredTolerance, declaredPassed});
            }
        }
    }
    const bool recoveryPassed = !native || (!modeChecks.empty()
        && std::all_of(modeChecks.begin(), modeChecks.end(), [](const auto& check) { return check.passed_m; }));
    std::vector<std::pair<std::string, bool>> checks = {
        {"particle_count", countPassed}, {"finite_wrapped_phase_space", phasePassed},
        {"unit_pm_weights", weightsPassed}, {"physical_particle_mass", physicalMassPassed},
        {"expected_id_signatures", idsPassed}, {"native_lagrangian_lattice", latticePassed},
        {"native_momentum_relation", momentumPassed}, {"native_finite_modes", finiteModesPassed},
        {"native_dc_zero", dcPassed}, {"native_cutoff_and_nyquist_zero", cutoffPassed},
        {"native_inverse_reality", realityPassed}, {"native_selected_mode_recovery", recoveryPassed},
        {"native_declared_mode_coefficients", !native || (!modeChecks.empty() && declaredModesPassed)}};
    // Most checks already use global reductions. Reduce the final decisions too
    // so a corrupted replicated scalar on one rank cannot return on its peers.
    std::vector<int> localDecisions(checks.size()), globalDecisions(checks.size());
    for (std::size_t i = 0; i < checks.size(); ++i) localDecisions[i] = checks[i].second ? 1 : 0;
    MPI_Allreduce(localDecisions.data(), globalDecisions.data(), static_cast<int>(checks.size()),
                  MPI_INT, MPI_MIN, communicator);
    for (std::size_t i = 0; i < checks.size(); ++i) checks[i].second = globalDecisions[i] != 0;
    const bool passed = std::all_of(checks.begin(), checks.end(), [](const auto& check) { return check.second; });
    int writeSucceeded = 1;
    if (rank == 0) {
        try {
            const auto output = std::filesystem::path(config_m.output);
            std::ofstream spectrum(output / "pk_initial.csv");
            spectrum << std::setprecision(17)
                << "# Native linear input field: D(a)^2 V |delta_0|^2; not particle CIC P(k); no shot-noise subtraction.\n";
            if (!native) spectrum << "# Not applicable: external phase space has no native linear input field.\n";
            spectrum << "shell_index,k_mean_h_mpc,modes_full,modes_independent,p_linear_initial_mpc_h_cubed,"
                        "p_target_initial_mpc_h_cubed,ratio_to_target,gaussian_fractional_sigma\n";
            for (const auto& shell : shells) {
                spectrum << shell.index_m << ',' << shell.waveNumber_m << ',' << shell.modes_m << ',' << shell.modes_m / 2
                    << ',' << shell.power_m << ',' << shell.target_m << ',';
                if (shell.target_m > 0) spectrum << shell.power_m / shell.target_m;
                spectrum << ',';
                if (gaussian) spectrum << std::sqrt(2 / double(shell.modes_m));
                spectrum << '\n';
            }
            spectrum.close();
            if (!spectrum) throw std::runtime_error("Cannot write initial linear spectrum");
            std::ofstream report(output / "ic_check.json.partial");
            report << std::setprecision(17) << std::boolalpha;
            report << "{\n  \"schema\":\"ippl-native-ic-check-v1\",\n  \"passed\":" << passed
                << ",\n  \"ic_mode\":\"" << config_m.icMode << "\",\n  \"native_1lpt\":" << native
                << ",\n  \"config\":{\"np\":" << n << ",\"seed\":" << config_m.seed
                << ",\"ic_rng\":\"" << config_m.icRng << "\",\"ic_mode_cutoff\":" << config_m.icModeCutoff
                << ",\"ic_momentum_precision\":\"" << config_m.icMomentumPrecision
                << "\",\"a_initial\":" << a << ",\"box_mpc_h\":" << box << ",\"ranks\":" << ranks << "},\n  \"checks\":{";
            for (std::size_t i = 0; i < checks.size(); ++i) {
                if (i) report << ',';
                report << "\n    \"" << checks[i].first << "\":";
                if (!native && checks[i].first.rfind("native_", 0) == 0) report << "null";
                else report << checks[i].second;
            }
            report << "\n  },\n  \"failures\":[";
            bool first = true;
            for (const auto& check : checks) if (!check.second) {
                if (!first) report << ',';
                report << '"' << check.first << '"';
                first = false;
            }
            report << "],\n  \"statistics\":{\"expected_particles\":" << expectedCount
                << ",\"observed_particles\":" << observedCount << ",\"invalid_phase_space_particles\":" << invariants[0]
                << ",\"invalid_pm_weights\":" << invariants[1] << ",\"out_of_range_ids\":" << invariants[2]
                << ",\"id_sum\":" << invariants[3] << ",\"expected_id_sum\":" << referenceSum
                << ",\"id_hash_sum\":" << invariants[4] << ",\"expected_id_hash_sum\":" << referenceHash
                << ",\"invalid_lagrangian_sites\":" << invariants[5] << ",\"physical_mass_ratio_to_cosmology\":";
            iccheck_detail::writeNumber(report, physicalMassRatio);
            report << ",\"maximum_momentum_error_over_tolerance\":";
            iccheck_detail::writeNumber(report, momentumErrorRatio);
            report << ",\"maximum_initial_displacement_mpc_h\":";
            iccheck_detail::writeNumber(report, maximumDisplacement);
            report << ",\"inverse_imaginary_max\":";
            iccheck_detail::writeNumber(report, maxImaginary_m);
            report << ",\"inverse_imaginary_tolerance\":";
            iccheck_detail::writeNumber(report, realityTolerance);
            report << ",\"gaussian_shells_outside_five_sigma\":" << spectrumWarnings << "},\n  \"selected_modes\":[";
            for (std::size_t i = 0; i < modeChecks.size(); ++i) {
                const auto& check = modeChecks[i];
                if (i) report << ',';
                report << "\n    {\"integer_mode\":[" << check.mode_m[0] << ',' << check.mode_m[1] << ',' << check.mode_m[2]
                    << "],\"expected_initial_real\":" << check.expectedReal_m
                    << ",\"expected_initial_imaginary\":" << check.expectedImaginary_m
                    << ",\"recovered_real\":" << check.recoveredReal_m << ",\"recovered_imaginary\":" << check.recoveredImaginary_m
                    << ",\"absolute_error\":" << check.error_m << ",\"tolerance\":" << check.tolerance_m
                    << ",\"passed\":" << check.passed_m
                    << ",\"declared_initial_real\":" << check.declaredReal_m
                    << ",\"declared_initial_imaginary\":" << check.declaredImaginary_m
                    << ",\"declared_absolute_error\":" << check.declaredError_m
                    << ",\"declared_tolerance\":" << check.declaredTolerance_m
                    << ",\"declared_passed\":" << check.declaredPassed_m << '}';
            }
            report << "\n  ],\n  \"notes\":[\"ID signatures are corruption checks, not an exact uniqueness proof.\","
                "\"The reported spectrum is the native linear field, not CIC particle density; no shot noise was subtracted.\","
                "\"Single-realization Gaussian shell variance is warning-only.\","
                "\"External phase space is not subjected to native 1LPT, cutoff or linear-spectrum tests.\"]\n}\n";
            report.close();
            if (!report) throw std::runtime_error("Cannot write initial-condition check");
            std::filesystem::rename(output / "ic_check.json.partial", output / "ic_check.json");
        } catch (const std::exception& error) {
            std::cerr << "Initial-condition report: " << error.what() << '\n';
            writeSucceeded = 0;
        }
    }
    MPI_Bcast(&writeSucceeded, 1, MPI_INT, 0, communicator);
    if (!writeSucceeded) throw std::runtime_error("Initial-condition acceptance report could not be written");
    if (!passed) throw std::runtime_error("Initial-condition acceptance failed; see ic_check.json");
}
} // namespace cosmology
#endif
