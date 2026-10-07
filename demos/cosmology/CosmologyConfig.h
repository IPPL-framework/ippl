/**
 * @brief Scientific implementation and contracts for CosmologyConfig.h.
 *
 * @file CosmologyConfig.h
 * @ingroup cosmology_core
 * @see cosmology_model cosmology_numerics cosmology_contracts
 */
#ifndef IPPL_COSMOLOGY_CONFIG_H
#define IPPL_COSMOLOGY_CONFIG_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cosmology {

/** Parameters shared by the initializer and evolution; length is comoving Mpc/h. */
/**
 * @brief Validated configuration shared by initialization and production evolution.
 *
 * The default model is flat matter plus Lambda, radiation-free and periodic.
 * All lengths are comoving Mpc/h. See @ref cosmology_contracts for keys/defaults.
 * This value object performs host-only work and owns no distributed state.
 */
struct Config {
    int nGrid = 32; ///< Force-mesh cells per dimension; production also generates nGrid^3 particles; input np.
    int nSteps = 100; ///< Positive number of uniform-log(a) evolution intervals; input nt.
    double boxSize = 200.0; ///< Positive comoving box side L in Mpc/h; input box_size.
    std::uint64_t seed = 1; ///< Full unsigned 64-bit production Fourier RNG seed.
    double zInitial = 49.0; ///< Initial redshift; must exceed zFinal; input z_in.
    double zFinal = 9.0; ///< Nonnegative final redshift; input z_fi.
    double hubble = 0.7; ///< Dimensionless h=H0/(100 km/s/Mpc); transfer-shape parameter.
    double omegaMatter = 0.3; ///< Total matter fraction at a=1; flat Lambda density is 1-omegaMatter.
    double omegaBaryon = 0.05; ///< Baryon contribution to the weighted transfer; no separate gas species is evolved.
    double sigma8 = 0.8; ///< Positive z=0 finite-cutoff top-hat RMS at R=8 Mpc/h.
    double spectralIndex = 0.96; ///< Primordial power-law index n_s in (0,2).
    int transferFunction = 4; ///< Transfer selector: 4 for BBKS, 0 for a CMBFAST-style table.
    std::string transferFile; ///< Transfer table path; relative paths are resolved from the parameter-file directory.
    std::string icMode = "gaussian"; ///< Initial state selector: gaussian, sine, uniform or external.
    std::string icRng = "legacy"; ///< Gaussian RNG: legacy mesh-indexed pairs or mode_hash_v1 resolution-independent physical modes.
    int icModeCutoff = 0; ///< Spherical integer-mode radius; 0 retains all non-Nyquist modes; positive values must be below np/2.
    std::string icMomentumPrecision = "double"; ///< Native 1LPT p precision: double, or one float32 rounding for saved-IC compatibility.
    bool icOnly = false; ///< Validate/export initial state with no KDK evolution; output_redshifts must be empty.
    std::string icFile; ///< Canonical binary phase-space input, resolved relative to the parameter file.
    std::uint64_t importedParticleCount = 0; ///< Explicit external count, independent of the force mesh; input particle_count.
    std::string snapshotFormat = "csv"; ///< Per-rank csv (legacy default) or binary phase-space snapshots.
    std::vector<double> outputRedshifts; ///< Strictly decreasing output epochs within [zFinal,zInitial).
    double amplitude = 1.0e-3; ///< Initial-redshift sine density amplitude; ignored for Gaussian/uniform initial states.
    std::array<int, 3> mode = {1, 0, 0}; ///< Signed integer sine-wave components, strictly below every Nyquist plane.
    std::string output = "cosmology-output"; ///< Output directory resolved from run working directory; nonempty existing simulation output is rejected.
    int diagnosticsEvery = 10; ///< Positive interval in steps between production diagnostic rows.
    bool writeParticles = true; ///< Whether production per-rank particle snapshots are written.

    /**
     * @brief Convert the configured starting redshift to scale factor.
     *
     * @pre validate() has accepted the configuration.
     * @return Dimensionless aInitial=1/(1+zInitial).
     */
    double aInitial() const { return 1.0 / (1.0 + zInitial); }
    /**
     * @brief Convert the configured final redshift to scale factor.
     *
     * @pre validate() has accepted the configuration.
     * @return Dimensionless aFinal=1/(1+zFinal).
     */
    double aFinal() const { return 1.0 / (1.0 + zFinal); }
    /**
     * @brief Count the production cubic particle load.
     *
     * @pre validate() has checked integer overflow. Imported evolution overrides the expected count separately.
     * @return Exact nGrid^3 as uint64.
     */
    std::uint64_t particleCount() const {
        return std::uint64_t(nGrid) * std::uint64_t(nGrid) * std::uint64_t(nGrid);
    }

    /**
     * @brief Reject unsupported physics and unsafe numerical/input ranges.
     *
     * @throws std::invalid_argument On an invalid value or unsupported model.
     * @see cosmology_contracts
     * Host-only; performs no MPI communication or I/O.
     */
    void validate() const {
        const auto require = [](bool valid, const std::string& message) {
            if (!valid) throw std::invalid_argument("Cosmology config: " + message);
        };
        require(nGrid >= 4 && nGrid % 2 == 0, "np must be even and at least 4");
        const auto grid = std::uint64_t(nGrid);
        require(grid <= std::numeric_limits<std::uint64_t>::max() / grid / grid,
                "np cubed exceeds the particle-count integer range");
        require(3 * (grid / 2) * (grid / 2) < std::uint64_t(std::numeric_limits<int>::max()),
                "np exceeds the Fourier squared-index integer range");
        require(nSteps > 0, "nt must be positive");
        require(std::isfinite(boxSize) && boxSize > 0.0, "box_size must be positive");
        require(std::isfinite(zInitial) && std::isfinite(zFinal) && zFinal >= 0.0
                    && zInitial > zFinal,
                "require z_in > z_fi >= 0");
        require(std::isfinite(hubble) && hubble > 0.0, "hubble must be positive");
        require(std::isfinite(omegaMatter) && omegaMatter > 0.0 && omegaMatter <= 1.0,
                "require 0 < Omega_m <= 1 (flat matter + cosmological constant)");
        require(std::isfinite(omegaBaryon) && omegaBaryon >= 0.0
                    && omegaBaryon <= omegaMatter, "require 0 <= Omega_bar <= Omega_m");
        require(std::isfinite(sigma8) && sigma8 > 0.0, "Sigma_8 must be positive");
        require(std::isfinite(spectralIndex) && spectralIndex > 0.0 && spectralIndex < 2.0,
                "this implementation supports 0 < n_s < 2");
        require(transferFunction == 0 || transferFunction == 4,
                "TFFlag must be 0 (CMBFAST table) or 4 (BBKS)");
        require(transferFunction != 0 || !transferFile.empty(),
                "transfer_file is required for TFFlag=0");
        require(icMode == "gaussian" || icMode == "sine" || icMode == "uniform" || icMode == "external",
                "ic_mode must be gaussian, sine, uniform, or external");
        require(icRng == "legacy" || icRng == "mode_hash_v1",
                "ic_rng must be legacy or mode_hash_v1");
        require(icModeCutoff >= 0 && icModeCutoff < nGrid / 2,
                "ic_mode_cutoff must be zero or a positive integer strictly below np/2");
        require(icMode == "gaussian" || (icRng == "legacy" && icModeCutoff == 0),
                "ic_rng and ic_mode_cutoff overrides apply only to Gaussian ICs");
        require(icMomentumPrecision == "double" || icMomentumPrecision == "float32",
                "ic_momentum_precision must be double or float32");
        require(icMode != "external" || icMomentumPrecision == "double",
                "external IC momenta are preserved; ic_momentum_precision must be double");
        require(!icOnly || outputRedshifts.empty(), "ic_only cannot request evolved output_redshifts");
        require(snapshotFormat == "csv" || snapshotFormat == "binary", "snapshot_format must be csv or binary");
        if (icMode == "external") {
            require(!icFile.empty() && importedParticleCount > 0,
                    "external IC requires ic_file and positive particle_count");
            require(importedParticleCount <= (std::numeric_limits<std::uint64_t>::max() - 128) / 56,
                    "particle_count exceeds the binary file size range");
        } else {
            require(icFile.empty() && importedParticleCount == 0,
                    "ic_file and particle_count apply only to ic_mode=external");
        }
        double previousRedshift = zInitial;
        for (double redshift : outputRedshifts) {
            require(std::isfinite(redshift) && redshift >= zFinal && redshift < previousRedshift,
                    "output_redshifts must decrease strictly within [z_fi,z_in)");
            const double previousA = 1 / (1 + previousRedshift), nextA = 1 / (1 + redshift);
            require(nextA - previousA > 16 * std::numeric_limits<double>::epsilon() * nextA,
                    "output epochs must be distinguishable at floating-point precision");
            require(redshift == zFinal || aFinal() - nextA > 16 * std::numeric_limits<double>::epsilon() * aFinal(),
                    "output epoch is indistinguishable from final epoch");
            previousRedshift = redshift;
        }
        require(std::isfinite(amplitude) && amplitude >= 0.0 && amplitude < 1.0,
                "amplitude must satisfy 0 <= amplitude < 1");
        if (icMode == "sine") {
            require(mode[0] != 0 || mode[1] != 0 || mode[2] != 0,
                    "the sine mode must be nonzero");
            for (const int component : mode)
                require(component > -nGrid / 2 && component < nGrid / 2,
                        "each sine-mode component must lie strictly below Nyquist");
        }
        require(!output.empty(), "output must name an output directory");
        require(diagnosticsEvery > 0, "diagnostics_every must be positive");
    }

    /**
     * @brief Union uniform-log(a) step endpoints with requested exact output epochs.
     * @return Increasing synchronized KDK endpoints, including aInitial and aFinal.
     * Requested epochs split intervals; nSteps counts the base intervals, not extra splits.
     * Near-identical endpoints (8 ulps relative) use the requested epoch to avoid zero steps.
     */
    std::vector<double> timePoints() const {
        std::vector<double> points;
        const double logStep = std::log(aFinal() / aInitial()) / nSteps;
        for (int step = 0; step <= nSteps; ++step)
            points.push_back(step == 0 ? aInitial() : step == nSteps ? aFinal()
                               : aInitial() * std::exp(step * logStep));
        for (const double redshift : outputRedshifts) {
            const double a = 1.0 / (1.0 + redshift);
            auto next = std::lower_bound(points.begin(), points.end(), a);
            const auto near = [a](double value) {
                return std::abs(value - a) <= 8 * std::numeric_limits<double>::epsilon() * a;
            };
            if (next != points.end() && near(*next)) *next = a;
            else if (next != points.begin() && near(*(next - 1))) *(next - 1) = a;
            else points.insert(next, a);
        }
        return points;
    }

    /**
     * @brief Parse, resolve and validate a cosmology parameter file.
     *
     * @throws std::runtime_error If the input file cannot be opened.
     * @throws std::invalid_argument For malformed/duplicate/unknown input or unsupported physics.
     * Host-only. Output paths remain relative to the eventual run directory.
     *
     * @param fileName Input parameter-file path; transfer paths are relative to its parent directory.
     * @return Validated configuration with resolved transfer path.
     */
    static Config fromFile(const std::string& fileName) {
        std::ifstream input(fileName);
        if (!input) throw std::runtime_error("Cannot open cosmology config: " + fileName);
        Config result;
        std::set<std::string> seen;
        std::string line;
        int lineNumber = 0;
        const auto trim = [](const std::string& text) {
            const auto first = text.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return std::string();
            return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
        };
        while (std::getline(input, line)) {
            ++lineNumber;
            line = line.substr(0, line.find("//"));
            line = trim(line.substr(0, line.find('#')));
            if (line.empty()) continue;
            const auto equals = line.find('=');
            const auto fail = [&](const std::string& message) {
                throw std::invalid_argument(fileName + ":" + std::to_string(lineNumber)
                                            + ": " + message);
            };
            if (equals == std::string::npos) fail("expected name=value");
            const std::string name = trim(line.substr(0, equals));
            std::string value = trim(line.substr(equals + 1));
            if (!seen.insert(name).second) fail("duplicate parameter " + name);
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
                value = value.substr(1, value.size() - 2);
            const auto number = [&](auto& target) {
                std::istringstream stream(value);
                stream >> target;
                if (!stream) fail("invalid numeric value for " + name);
                stream >> std::ws;
                if (!stream.eof()) fail("trailing characters for " + name);
            };
            if (name == "np") number(result.nGrid);
            else if (name == "nt") number(result.nSteps);
            else if (name == "box_size") number(result.boxSize);
            else if (name == "seed") {
                if (value.empty() || value.front() == '-') fail("seed must be unsigned");
                number(result.seed);
            }
            else if (name == "z_in") number(result.zInitial);
            else if (name == "z_fi") number(result.zFinal);
            else if (name == "hubble") number(result.hubble);
            else if (name == "Omega_m") number(result.omegaMatter);
            else if (name == "Omega_bar") number(result.omegaBaryon);
            else if (name == "Sigma_8") number(result.sigma8);
            else if (name == "n_s") number(result.spectralIndex);
            else if (name == "TFFlag") number(result.transferFunction);
            else if (name == "transfer_file") result.transferFile = value;
            else if (name == "ic_mode") result.icMode = value;
            else if (name == "ic_rng") result.icRng = value;
            else if (name == "ic_mode_cutoff") number(result.icModeCutoff);
            else if (name == "ic_momentum_precision") result.icMomentumPrecision = value;
            else if (name == "ic_only") {
                if (value == "true" || value == "1") result.icOnly = true;
                else if (value == "false" || value == "0") result.icOnly = false;
                else fail("ic_only must be true, false, 1, or 0");
            }
            else if (name == "ic_file") result.icFile = value;
            else if (name == "particle_count") {
                if (value.empty() || value.front() == '-') fail("particle_count must be unsigned");
                number(result.importedParticleCount);
            }
            else if (name == "snapshot_format") result.snapshotFormat = value;
            else if (name == "output_redshifts") {
                if (value.empty() || value.back() == ',') fail("output_redshifts requires a nonempty comma-separated list");
                std::istringstream list(value);
                std::string item;
                while (std::getline(list, item, ',')) {
                    std::istringstream entry(item);
                    double redshift;
                    if (!(entry >> redshift)) fail("invalid output redshift");
                    entry >> std::ws;
                    if (!entry.eof()) fail("trailing characters in output redshift");
                    result.outputRedshifts.push_back(redshift);
                }
            }
            else if (name == "amplitude") number(result.amplitude);
            else if (name == "mode_x") number(result.mode[0]);
            else if (name == "mode_y") number(result.mode[1]);
            else if (name == "mode_z") number(result.mode[2]);
            else if (name == "output") result.output = value;
            else if (name == "diagnostics_every") number(result.diagnosticsEvery);
            else if (name == "write_particles") {
                if (value == "true" || value == "1") result.writeParticles = true;
                else if (value == "false" || value == "0") result.writeParticles = false;
                else fail("write_particles must be true, false, 1, or 0");
            }
            else if (name == "Omega_nu" || name == "Omega_r" || name == "f_NL"
                     || name == "w_de") {
                double unsupported = 0.0;
                number(unsupported);
                const double required = name == "w_de" ? -1.0 : 0.0;
                if (unsupported != required)
                    fail("only " + name + "=" + std::to_string(required) + " is supported");
            }
            else fail("unknown parameter " + name);
        }
        if (!result.transferFile.empty()) {
            std::filesystem::path transferPath(result.transferFile);
            if (transferPath.is_relative())
                transferPath = std::filesystem::path(fileName).parent_path() / transferPath;
            result.transferFile = transferPath.lexically_normal().string();
        }
        if (!result.icFile.empty()) {
            std::filesystem::path inputPath(result.icFile);
            if (inputPath.is_relative()) inputPath = std::filesystem::path(fileName).parent_path() / inputPath;
            result.icFile = inputPath.lexically_normal().string();
        }
        result.validate();
        return result;
    }
};

}  // namespace cosmology

#endif
