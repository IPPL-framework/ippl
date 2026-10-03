#ifndef IPPL_COSMOLOGY_CONFIG_H
#define IPPL_COSMOLOGY_CONFIG_H

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

namespace cosmology {

/** Parameters shared by the initializer and evolution; length is comoving Mpc/h. */
struct Config {
    int nGrid = 32;
    int nSteps = 100;
    double boxSize = 200.0;
    std::uint64_t seed = 1;
    double zInitial = 49.0;
    double zFinal = 9.0;
    double hubble = 0.7;
    double omegaMatter = 0.3;
    double omegaBaryon = 0.05;
    double sigma8 = 0.8;
    double spectralIndex = 0.96;
    int transferFunction = 4;
    std::string transferFile;
    std::string icMode = "gaussian";
    double amplitude = 1.0e-3;
    std::array<int, 3> mode = {1, 0, 0};
    std::string output = "cosmology-output";
    int diagnosticsEvery = 10;
    bool writeParticles = true;

    double aInitial() const { return 1.0 / (1.0 + zInitial); }
    double aFinal() const { return 1.0 / (1.0 + zFinal); }
    std::uint64_t particleCount() const {
        return std::uint64_t(nGrid) * std::uint64_t(nGrid) * std::uint64_t(nGrid);
    }

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
        require(icMode == "gaussian" || icMode == "sine" || icMode == "uniform",
                "ic_mode must be gaussian, sine, or uniform");
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
        result.validate();
        return result;
    }
};

}  // namespace cosmology

#endif
