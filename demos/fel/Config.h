#ifndef IPPL_FEL_CONFIG_H
#define IPPL_FEL_CONFIG_H

// Configuration parsing for the FEL simulation.
//
// Reads a MITHRA-style JSON job file and converts all physical quantities into
// the natural unit system defined in units.h. Ported from the original
// FreeElectronLaser.cpp.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "Types/Vector.h"

#include "units.h"

#define JSON_HAS_RANGES 0  
#include <json.hpp>

struct config {
    using scalar = double;

    // GRID PARAMETERS
    ippl::Vector<uint32_t, 3> resolution;  // Grid resolution in 3D
    ippl::Vector<scalar, 3> extents;       // Physical extents of the grid in each dimension
    scalar total_time;                     // Total simulation time
    scalar timestep_ratio;                 // Ratio of timestep to some reference value

    scalar length_scale_in_jobfile;    // Length scale defined in the jobfile
    scalar temporal_scale_in_jobfile;  // Temporal scale defined in the jobfile

    // PARTICLE PARAMETERS
    scalar charge;           // Particle charge in unit_charge
    scalar mass;             // Particle mass in unit_mass
    uint64_t num_particles;  // Number of particles in the simulation
    bool space_charge;       // Flag for considering space charge effects

    // BUNCH PARAMETERS
    ippl::Vector<scalar, 3> mean_position;   // Mean initial position of the particle bunch
    ippl::Vector<scalar, 3> sigma_position;  // Standard deviation of the initial position distribution
    ippl::Vector<scalar, 3> position_truncations;  // Truncations of the position distribution
    ippl::Vector<scalar, 3> sigma_momentum;  // Standard deviation of the initial momentum distribution
    scalar bunch_gamma;                      // Relativistic gamma factor of the bunch

    // UNDULATOR PARAMETERS
    scalar undulator_K;       // Undulator parameter K
    scalar undulator_period;  // Period of the undulator
    scalar undulator_length;  // Length of the undulator

    uint32_t output_rhythm;                                      // Frequency of output in timesteps
    std::string output_path;                                     // Path to output files
    std::unordered_map<std::string, double> experiment_options;  // Additional experimental options
};

template <typename scalar, unsigned Dim>
ippl::Vector<scalar, Dim> getVector(const nlohmann::json& j) {
    if (j.is_array()) {
        assert(j.size() == Dim);
        ippl::Vector<scalar, Dim> ret;
        for (unsigned i = 0; i < Dim; i++)
            ret[i] = (scalar)j[i];
        return ret;
    } else {
        std::cerr << "Warning: Obtaining Vector from scalar json\n";
        ippl::Vector<scalar, Dim> ret = (scalar)j;
        return ret;
    }
}

// Compile-time / run-time string hashing used to switch on unit-scale names.
template <size_t N, typename T>
struct DefaultedStringLiteral {
    constexpr DefaultedStringLiteral(const char (&str)[N], const T val)
        : value(val) {
        std::copy_n(str, N, key);
    }

    T value;
    char key[N];
};
template <size_t N>
struct StringLiteral {
    constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }

    char value[N];
    constexpr DefaultedStringLiteral<N, int> operator>>(int t) const noexcept {
        return DefaultedStringLiteral<N, int>(value, t);
    }
    constexpr size_t size() const noexcept { return N - 1; }
};
template <StringLiteral lit>
constexpr size_t chash() {
    size_t hash = 5381;
    int c;

    for (size_t i = 0; i < lit.size(); i++) {
        c    = lit.value[i];
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }

    return hash;
}
inline size_t chash(const char* val) {
    size_t hash = 5381;
    int c;

    while ((c = *val++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }

    return hash;
}
inline size_t chash(const std::string& _val) {
    size_t hash     = 5381;
    const char* val = _val.c_str();
    int c;

    while ((c = *val++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }

    return hash;
}
inline std::string lowercase_singular(std::string str) {
    // Convert string to lowercase
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);

    // Check if the string ends with "s" and remove it if it does
    if (!str.empty() && str.back() == 's') {
        str.pop_back();
    }

    return str;
}
inline double get_time_multiplier(const nlohmann::json& j) {
    std::string length_scale_string = lowercase_singular((std::string)j["mesh"]["time-scale"]);
    double time_factor              = 1.0;
    switch (chash(length_scale_string)) {
        case chash<"planck-time">():
        case chash<"plancktime">():
        case chash<"pt">():
        case chash<"natural">():
            time_factor = unit_time_in_seconds;
            break;
        case chash<"picosecond">():
            time_factor = 1e-12;
            break;
        case chash<"nanosecond">():
            time_factor = 1e-9;
            break;
        case chash<"microsecond">():
            time_factor = 1e-6;
            break;
        case chash<"millisecond">():
            time_factor = 1e-3;
            break;
        case chash<"second">():
            time_factor = 1.0;
            break;
        default:
            std::cerr << "Unrecognized time scale: " << (std::string)j["mesh"]["time-scale"]
                      << "\n";
            break;
    }
    return time_factor;
}
inline double get_length_multiplier(const nlohmann::json& options) {
    std::string length_scale_string =
        lowercase_singular((std::string)options["mesh"]["length-scale"]);
    double length_factor = 1.0;
    switch (chash(length_scale_string)) {
        case chash<"planck-length">():
        case chash<"plancklength">():
        case chash<"pl">():
        case chash<"natural">():
            length_factor = unit_length_in_meters;
            break;
        case chash<"picometer">():
            length_factor = 1e-12;
            break;
        case chash<"nanometer">():
            length_factor = 1e-9;
            break;
        case chash<"micrometer">():
            length_factor = 1e-6;
            break;
        case chash<"millimeter">():
            length_factor = 1e-3;
            break;
        case chash<"meter">():
            length_factor = 1.0;
            break;
        default:
            std::cerr << "Unrecognized length scale: "
                      << (std::string)options["mesh"]["length-scale"] << "\n";
            break;
    }
    return length_factor;
}
inline config read_config(const char* filepath) {
    std::ifstream cfile(filepath);
    nlohmann::json j;
    cfile >> j;
    config::scalar lmult = get_length_multiplier(j);
    config::scalar tmult = get_time_multiplier(j);
    config ret;

    ret.extents[0] = ((config::scalar)j["mesh"]["extents"][0] * lmult) / unit_length_in_meters;
    ret.extents[1] = ((config::scalar)j["mesh"]["extents"][1] * lmult) / unit_length_in_meters;
    ret.extents[2] = ((config::scalar)j["mesh"]["extents"][2] * lmult) / unit_length_in_meters;
    ret.resolution = getVector<uint32_t, 3>(j["mesh"]["resolution"]);

    if (j.contains("timestep-ratio")) {
        ret.timestep_ratio = (config::scalar)j["timestep-ratio"];
    } else {
        ret.timestep_ratio = 1;
    }
    ret.total_time   = ((config::scalar)j["mesh"]["total-time"] * tmult) / unit_time_in_seconds;
    ret.space_charge = (bool)(j["mesh"]["space-charge"]);
    ret.bunch_gamma  = (config::scalar)(j["bunch"]["gamma"]);
    if (ret.bunch_gamma < config::scalar(1)) {
        std::cerr << "Gamma must be >= 1\n";
        exit(1);
    }
    assert(j.contains("undulator"));
    assert(j["undulator"].contains("static-undulator"));

    ret.undulator_K      = j["undulator"]["static-undulator"]["undulator-parameter"];
    ret.undulator_period = ((config::scalar)j["undulator"]["static-undulator"]["period"] * lmult)
                           / unit_length_in_meters;
    ret.undulator_length = ((config::scalar)j["undulator"]["static-undulator"]["length"] * lmult)
                           / unit_length_in_meters;
    assert(!std::isnan(ret.undulator_length));
    assert(!std::isnan(ret.undulator_period));
    assert(!std::isnan(ret.extents[0]));
    assert(!std::isnan(ret.extents[1]));
    assert(!std::isnan(ret.extents[2]));
    assert(!std::isnan(ret.total_time));
    ret.length_scale_in_jobfile   = get_length_multiplier(j);
    ret.temporal_scale_in_jobfile = get_time_multiplier(j);
    ret.charge        = (config::scalar)j["bunch"]["charge"] * electron_charge_in_unit_charges;
    ret.mass          = (config::scalar)j["bunch"]["mass"] * electron_mass_in_unit_masses;
    ret.num_particles = (uint64_t)j["bunch"]["number-of-particles"];
    ret.mean_position =
        getVector<config::scalar, 3>(j["bunch"]["position"]) * lmult / unit_length_in_meters;
    ret.sigma_position =
        getVector<config::scalar, 3>(j["bunch"]["sigma-position"]) * lmult / unit_length_in_meters;
    ret.position_truncations = getVector<config::scalar, 3>(j["bunch"]["distribution-truncations"])
                               * lmult / unit_length_in_meters;
    ret.sigma_momentum = getVector<config::scalar, 3>(j["bunch"]["sigma-momentum"]);
    ret.output_rhythm  = j["output"].contains("rhythm") ? uint32_t(j["output"]["rhythm"]) : 0;
    ret.output_path    = "../data/";
    if (j["output"].contains("path")) {
        ret.output_path = j["output"]["path"];
        if (!ret.output_path.ends_with('/')) {
            ret.output_path.push_back('/');
        }
    }
    if (j.contains("experimentation")) {
        nlohmann::json je = j["experimentation"];
        for (auto it = je.begin(); it != je.end(); it++) {
            ret.experiment_options[it.key()] = double(it.value());
        }
    }
    return ret;
}

#endif
