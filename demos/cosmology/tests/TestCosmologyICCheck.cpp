/**
 * @file TestCosmologyICCheck.cpp
 * @brief Positive and deliberately corrupted inputs through the actual production IC gate.
 * @ingroup cosmology_diagnostics
 * @see cosmology_validation cosmology::Simulation::validateInitialConditions
 *
 * This regression generates small native inputs with production initialization,
 * changes only test-owned state through Simulation's test friend, then invokes
 * the production acceptance gate. No force solve, KDK step or production fault
 * injection switch is used. Each failed case must be rejected on every rank,
 * publish its expected failed check, and leave the particle force at zero.
 */
#include "CosmologySimulation.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace cosmology {

/**
 * @brief Test-only friend controlling initialized state before the production acceptance gate.
 *
 * Every mutation is confined to this executable's local Simulation objects.
 * Production has no runtime fault selector and no behavior conditional on tests.
 */
struct InitialConditionTest {
    /** @brief Input corruption applied before calling the real gate. */
    enum class Fault {
        None, UnitWeight, PhysicalMass, NonfiniteMomentum, DuplicateId,
        MissingParticle, MomentumRelation, DeclaredSeed, DeclaredCutoff
    };

    /**
     * @brief Require a test assertion collectively to avoid divergent rank cleanup.
     * @param condition Local success state.
     * @param message Failure explanation printed before communicator abort.
     */
    static void requireAll(bool condition, const std::string& message) {
        const int local = condition ? 1 : 0;
        int all = 0;
        MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, ippl::Comm->getCommunicator());
        if (!all) throw std::runtime_error(message);
    }

    /**
     * @brief Create one fresh test receipt directory collectively.
     * @param path New child of the unique root selected by rank zero.
     */
    static void createDirectory(const std::filesystem::path& path) {
        int created = 1;
        if (ippl::Comm->rank() == 0) {
            try {
                created = std::filesystem::create_directories(path) ? 1 : 0;
            } catch (const std::exception& error) {
                std::cerr << error.what() << '\n';
                created = 0;
            }
        }
        MPI_Bcast(&created, 1, MPI_INT, 0, ippl::Comm->getCommunicator());
        if (!created) throw std::runtime_error("Cannot create a fresh gate-test directory: " + path.string());
        ippl::Comm->barrier();
    }

    /**
     * @brief Produce the native state using exactly the production initialization calls.
     * @param simulation Fresh Simulation with valid native configuration.
     * @param fault Optionally construct a field inconsistent with its declared seed/cutoff.
     */
    static void initialize(Simulation& simulation, Fault fault) {
        simulation.createLattice();
        const auto seed = simulation.config_m.seed;
        const auto cutoff = simulation.config_m.icModeCutoff;
        if (fault == Fault::DeclaredSeed) simulation.config_m.seed ^= UINT64_C(0x123456789abcdef0);
        if (fault == Fault::DeclaredCutoff) simulation.config_m.icModeCutoff = cutoff + 1;
        simulation.initializeModes();
        simulation.displaceParticles();
        simulation.config_m.seed = seed;
        simulation.config_m.icModeCutoff = cutoff;
        simulation.particleMass_m = 2.77536627e11 * simulation.config_m.omegaMatter
            * std::pow(simulation.config_m.boxSize, 3) / simulation.totalParticles();
        simulation.verifyParticleLabels(true);
    }

    /**
     * @brief Corrupt actual local particle storage, selecting globally unique ID zero.
     * @param simulation Test-owned initialized state.
     * @param fault Specific scalar, label or count invariant to invalidate.
     */
    static void corrupt(Simulation& simulation, Fault fault) {
        if (fault == Fault::PhysicalMass) {
            simulation.particleMass_m *= 2;
            return;
        }
        if (fault == Fault::None || fault == Fault::DeclaredSeed || fault == Fault::DeclaredCutoff) return;
        const auto count = simulation.particles_m->getLocalNum();
        const auto ids = simulation.particles_m->globalId.getView();
        const auto weights = simulation.particles_m->mass.getView();
        const auto momenta = simulation.particles_m->momentum.getView();
        const double invalid = std::numeric_limits<double>::quiet_NaN();
        Kokkos::View<bool*> removed("IC gate test removed-particle mask", count);
        std::size_t selected = 0;
        Kokkos::parallel_reduce("IC gate test corrupt one particle", count,
            KOKKOS_LAMBDA(const std::size_t i, std::size_t& total) {
                const bool chosen = ids(i) == 0;
                removed(i) = chosen;
                total += chosen;
                if (!chosen) return;
                if (fault == Fault::UnitWeight) weights(i) = 2;
                if (fault == Fault::NonfiniteMomentum) momenta(i)[0] = invalid;
                if (fault == Fault::DuplicateId) ids(i) = 1;
                if (fault == Fault::MomentumRelation) momenta(i)[0] += 1.e-3;
            }, selected);
        const std::uint64_t selected64 = selected;
        std::uint64_t globalSelected = 0;
        MPI_Allreduce(&selected64, &globalSelected, 1, MPI_UINT64_T, MPI_SUM,
                      ippl::Comm->getCommunicator());
        requireAll(globalSelected == 1, "Fault injection must select exactly one actual particle");
        if (fault == Fault::MissingParticle) {
            // destroy is collective even on ranks whose local selection is empty.
            simulation.particles_m->destroy(removed, selected);
        }
    }

    /**
     * @brief Confirm no force solve has occurred while exercising the initialization gate.
     * @param simulation State after a gate pass or rejection.
     */
    static void checkUntouchedForces(const Simulation& simulation) {
        const auto force = simulation.particles_m->force.getView();
        std::uint64_t nonzero = 0;
        Kokkos::parallel_reduce("IC gate test force untouched", simulation.particles_m->getLocalNum(),
            KOKKOS_LAMBDA(const std::size_t i, std::uint64_t& bad) {
                for (int d = 0; d < 3; ++d) bad += (force(i)[d] != 0.0);
            }, nonzero);
        requireAll(nonzero == 0, "Initial-condition gate changed the untouched zero particle forces");
    }

    /**
     * @brief Run one pass/reject case and require its published receipt to agree.
     * @param root Unique test artifact directory; existing evidence is never overwritten.
     * @param name Stable case directory and diagnostic label.
     * @param mode Native mode selector.
     * @param rng Gaussian RNG selector; legacy for sine/uniform.
     * @param precision Initial momentum rounding contract.
     * @param fault Test-only perturbation; None requires acceptance.
     * @param expectedFailure Gate key that must explicitly be false after rejection.
     */
    static void runCase(const std::filesystem::path& root, const std::string& name,
                        const std::string& mode, const std::string& rng,
                        const std::string& precision, Fault fault,
                        const std::string& expectedFailure = "") {
        Config config;
        config.nGrid = 16;
        config.nSteps = 1;
        config.boxSize = 168.75;
        config.zInitial = 99;
        config.zFinal = 0;
        config.seed = UINT64_C(73452342811);
        config.icMode = mode;
        config.icRng = rng;
        config.icModeCutoff = mode == "gaussian" ? 3 : 0;
        config.icMomentumPrecision = precision;
        config.icOnly = true;
        config.output = (root / name).string();
        createDirectory(config.output);
        Simulation simulation(config);
        // Fatal test failures abort while the collective Simulation owners live.
        try {
            initialize(simulation, fault);
            corrupt(simulation, fault);
            bool rejected = false;
            std::string rejection;
            try {
                simulation.validateInitialConditions();
            } catch (const std::runtime_error& error) {
                rejected = true;
                rejection = error.what();
            }
            const bool expectedRejection = fault != Fault::None;
            requireAll(rejected == expectedRejection,
                       name + ": production gate acceptance differs from expected result: " + rejection);
            checkUntouchedForces(simulation);
            bool receiptPassed = true;
            if (ippl::Comm->rank() == 0) {
                std::ifstream stream(std::filesystem::path(config.output) / "ic_check.json");
                std::string report((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
                report.erase(std::remove_if(report.begin(), report.end(), [](unsigned char value) {
                    return std::isspace(value);
                }), report.end());
                const auto acceptance = expectedRejection ? "\"passed\":false" : "\"passed\":true";
                receiptPassed = bool(stream) && report.find("\"schema\":\"ippl-native-ic-check-v1\"")
                    != std::string::npos && report.find(acceptance) != std::string::npos;
                if (expectedRejection)
                    receiptPassed = receiptPassed && report.find("\"" + expectedFailure + "\":false")
                        != std::string::npos;
                for (const auto artifact : {"diagnostics.csv", "snapshots.csv", "metadata.txt"})
                    receiptPassed = receiptPassed && !std::filesystem::exists(std::filesystem::path(config.output) / artifact);
            }
            requireAll(receiptPassed, name + ": missing/wrong gate receipt or unexpected evolution artifact");
            if (ippl::Comm->rank() == 0)
                std::cout << "PASS " << name << ": " << (expectedRejection ? "collectively rejected" : "accepted")
                          << " before force/evolution; " << config.output << '\n';
        } catch (const std::exception& error) {
            std::cerr << "IC gate test rank " << ippl::Comm->rank() << ": " << error.what() << '\n';
            MPI_Abort(ippl::Comm->getCommunicator(), 1);
        }
    }

    /** @brief Exercise valid paths and independent corruptions through production code. */
    static void run() {
        std::string root;
        if (ippl::Comm->rank() == 0) {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            root = (std::filesystem::temp_directory_path() /
                    ("ippl-ic-gate-test-" + std::to_string(stamp))).string();
        }
        int length = static_cast<int>(root.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, ippl::Comm->getCommunicator());
        root.resize(length);
        MPI_Bcast(root.data(), length, MPI_CHAR, 0, ippl::Comm->getCommunicator());
        createDirectory(root);
        runCase(root, "uniform", "uniform", "legacy", "double", Fault::None);
        runCase(root, "sine", "sine", "legacy", "double", Fault::None);
        runCase(root, "gaussian_legacy", "gaussian", "legacy", "double", Fault::None);
        runCase(root, "gaussian_mode_hash", "gaussian", "mode_hash_v1", "double", Fault::None);
        runCase(root, "gaussian_mode_hash_float32", "gaussian", "mode_hash_v1", "float32", Fault::None);
        runCase(root, "bad_unit_weight", "gaussian", "mode_hash_v1", "double",
                Fault::UnitWeight, "unit_pm_weights");
        runCase(root, "bad_physical_mass", "gaussian", "mode_hash_v1", "double",
                Fault::PhysicalMass, "physical_particle_mass");
        runCase(root, "nonfinite_momentum", "gaussian", "mode_hash_v1", "double",
                Fault::NonfiniteMomentum, "finite_wrapped_phase_space");
        runCase(root, "duplicate_id", "gaussian", "mode_hash_v1", "double",
                Fault::DuplicateId, "expected_id_signatures");
        runCase(root, "missing_particle", "gaussian", "mode_hash_v1", "double",
                Fault::MissingParticle, "particle_count");
        runCase(root, "wrong_momentum", "gaussian", "mode_hash_v1", "double",
                Fault::MomentumRelation, "native_momentum_relation");
        runCase(root, "wrong_declared_seed", "gaussian", "mode_hash_v1", "double",
                Fault::DeclaredSeed, "native_declared_mode_coefficients");
        runCase(root, "wrong_declared_cutoff", "gaussian", "mode_hash_v1", "double",
                Fault::DeclaredCutoff, "native_cutoff_and_nyquist_zero");
        if (ippl::Comm->rank() == 0)
            std::cout << "PASS: 5 valid IC paths and 8 deliberately invalid inputs; ranks="
                      << ippl::Comm->size() << "; receipts=" << root << '\n';
    }
};

} // namespace cosmology

/**
 * @brief Run collective acceptance tests using the configured CPU/GPU IPPL backend.
 * @param argc Process argument count forwarded to IPPL initialization.
 * @param argv Process arguments forwarded to IPPL initialization.
 * @return Zero only when every valid and invalid case has its expected outcome.
 */
int main(int argc, char** argv) {
    ippl::initialize(argc, argv);
    try {
        cosmology::InitialConditionTest::run();
    } catch (const std::exception& error) {
        std::cerr << "IC gate test rank " << ippl::Comm->rank() << ": " << error.what() << '\n';
        MPI_Abort(ippl::Comm->getCommunicator(), 1);
        return 1;
    }
    ippl::finalize();
    return 0;
}
