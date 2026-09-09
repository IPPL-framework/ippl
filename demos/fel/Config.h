#ifndef IPPL_FEL_CONFIG_H
#define IPPL_FEL_CONFIG_H

// Configuration parsing for the FEL simulation.
//
// Reads a MITHRA-style JSON job file and converts all physical quantities into
// the natural unit system defined in units.h. Ported from the original
// FreeElectronLaser.cpp.

#include <algorithm>
#include <catalyst_conduit.hpp>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "Types/Vector.h"

#include "units.h"

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
    ippl::Vector<scalar, 3> mean_position;  // Mean initial position of the particle bunch
    ippl::Vector<scalar, 3>
        sigma_position;  // Standard deviation of the initial position distribution
    ippl::Vector<scalar, 3> position_truncations;  // Truncations of the position distribution
    ippl::Vector<scalar, 3>
        sigma_momentum;  // Standard deviation of the initial momentum distribution
    scalar bunch_gamma;  // Relativistic gamma factor of the bunch

    // UNDULATOR PARAMETERS
    scalar undulator_K;       // Undulator parameter K
    scalar undulator_period;  // Period of the undulator
    scalar undulator_length;  // Length of the undulator

    std::string output_path;                                     // Path to output files
    std::unordered_map<std::string, double> experiment_options;  // Additional experimental options
};

namespace fel_config_detail {

    inline conduit_cpp::Node requiredNode(const conduit_cpp::Node& root, const std::string& path) {
        if (!root.has_path(path)) {
            throw std::runtime_error("Missing required configuration value '" + path + "'");
        }
        return root[path];
    }

    inline long double numericElement(const conduit_cpp::Node& node, conduit_index_t index,
                                      const std::string& path) {
        if (!node.dtype().is_number() || index < 0 || index >= node.number_of_elements()) {
            throw std::runtime_error("Configuration value '" + path + "' must be numeric");
        }

        using Id = conduit_cpp::DataType::Id;
        switch (node.dtype().id()) {
            case Id::int8:
                return node.as_int8_ptr()[index];
            case Id::int16:
                return node.as_int16_ptr()[index];
            case Id::int32:
                return node.as_int32_ptr()[index];
            case Id::int64:
                return static_cast<long double>(node.as_int64_ptr()[index]);
            case Id::uint8:
                return node.as_uint8_ptr()[index];
            case Id::uint16:
                return node.as_uint16_ptr()[index];
            case Id::uint32:
                return node.as_uint32_ptr()[index];
            case Id::uint64:
                return static_cast<long double>(node.as_uint64_ptr()[index]);
            case Id::float32:
                return node.as_float32_ptr()[index];
            case Id::float64:
                return node.as_float64_ptr()[index];
            case Id::unknown:
                break;
        }

        throw std::runtime_error("Unsupported numeric type for configuration value '" + path + "'");
    }

    inline double requiredNumber(const conduit_cpp::Node& root, const std::string& path) {
        const auto node = requiredNode(root, path);
        if (node.number_of_elements() != 1) {
            throw std::runtime_error("Configuration value '" + path + "' must be a scalar");
        }
        return static_cast<double>(numericElement(node, 0, path));
    }

    inline std::string requiredString(const conduit_cpp::Node& root, const std::string& path) {
        const auto node = requiredNode(root, path);
        if (!node.dtype().is_string()) {
            throw std::runtime_error("Configuration value '" + path + "' must be a string");
        }
        return node.as_string();
    }

    template <typename Scalar>
    Scalar checkedNumericCast(long double value, const std::string& path) {
        if constexpr (std::is_integral_v<Scalar>) {
            if (!std::isfinite(value) || std::trunc(value) != value
                || value < static_cast<long double>(std::numeric_limits<Scalar>::lowest())
                || value > static_cast<long double>(std::numeric_limits<Scalar>::max())) {
                throw std::runtime_error("Configuration value '" + path
                                         + "' must be an in-range integer");
            }
        }
        return static_cast<Scalar>(value);
    }

    template <typename Scalar>
    Scalar requiredInteger(const conduit_cpp::Node& root, const std::string& path) {
        static_assert(std::is_integral_v<Scalar>);
        const auto node = requiredNode(root, path);
        if (node.number_of_elements() != 1) {
            throw std::runtime_error("Configuration value '" + path + "' must be a scalar");
        }
        return checkedNumericCast<Scalar>(numericElement(node, 0, path), path);
    }

    template <typename Scalar, unsigned Dim>
    ippl::Vector<Scalar, Dim> getVector(const conduit_cpp::Node& root, const std::string& path) {
        const auto node = requiredNode(root, path);
        if (!node.dtype().is_number()) {
            throw std::runtime_error("Configuration value '" + path + "' must be numeric");
        }

        const auto size = node.number_of_elements();
        ippl::Vector<Scalar, Dim> result;
        if (size == 1) {
            std::cerr << "Warning: Obtaining vector from scalar configuration value '" << path
                      << "'\n";
            result = checkedNumericCast<Scalar>(numericElement(node, 0, path), path);
            return result;
        }
        if (size != static_cast<conduit_index_t>(Dim)) {
            throw std::runtime_error("Configuration value '" + path + "' must contain "
                                     + std::to_string(Dim) + " elements");
        }

        for (unsigned i = 0; i < Dim; ++i) {
            result[i] = checkedNumericCast<Scalar>(numericElement(node, i, path), path);
        }
        return result;
    }

}  // namespace fel_config_detail

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
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    // Check if the string ends with "s" and remove it if it does
    if (!str.empty() && str.back() == 's') {
        str.pop_back();
    }

    return str;
}
inline double get_time_multiplier(const conduit_cpp::Node& root) {
    const std::string time_scale  = fel_config_detail::requiredString(root, "mesh/time-scale");
    std::string time_scale_string = lowercase_singular(time_scale);
    double time_factor            = 1.0;
    switch (chash(time_scale_string)) {
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
            std::cerr << "Unrecognized time scale: " << time_scale << "\n";
            break;
    }
    return time_factor;
}
inline double get_length_multiplier(const conduit_cpp::Node& root) {
    const std::string length_scale  = fel_config_detail::requiredString(root, "mesh/length-scale");
    std::string length_scale_string = lowercase_singular(length_scale);
    double length_factor            = 1.0;
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
            std::cerr << "Unrecognized length scale: " << length_scale << "\n";
            break;
    }
    return length_factor;
}
inline config read_config(const char* filepath) {
    try {
        conduit_cpp::Node root;
        conduit_node_load(conduit_cpp::c_node(&root), filepath, "json");

        const config::scalar lmult = get_length_multiplier(root);
        const config::scalar tmult = get_time_multiplier(root);
        config ret{};

        ret.extents = fel_config_detail::getVector<config::scalar, 3>(root, "mesh/extents") * lmult
                      / unit_length_in_meters;
        ret.resolution = fel_config_detail::getVector<uint32_t, 3>(root, "mesh/resolution");

        ret.timestep_ratio = root.has_path("timestep-ratio")
                                 ? fel_config_detail::requiredNumber(root, "timestep-ratio")
                                 : config::scalar(1);
        ret.total_time     = fel_config_detail::requiredNumber(root, "mesh/total-time") * tmult
                             / unit_time_in_seconds;
        ret.space_charge =
            fel_config_detail::requiredNumber(root, "mesh/space-charge") != config::scalar(0);
        ret.bunch_gamma = fel_config_detail::requiredNumber(root, "bunch/gamma");
        if (ret.bunch_gamma < config::scalar(1)) {
            throw std::runtime_error("Configuration value 'bunch/gamma' must be >= 1");
        }

        ret.undulator_K = fel_config_detail::requiredNumber(
            root, "undulator/static-undulator/undulator-parameter");
        ret.undulator_period =
            fel_config_detail::requiredNumber(root, "undulator/static-undulator/period") * lmult
            / unit_length_in_meters;
        ret.undulator_length =
            fel_config_detail::requiredNumber(root, "undulator/static-undulator/length") * lmult
            / unit_length_in_meters;

        if (!std::isfinite(ret.undulator_length) || !std::isfinite(ret.undulator_period)
            || !std::isfinite(ret.extents[0]) || !std::isfinite(ret.extents[1])
            || !std::isfinite(ret.extents[2]) || !std::isfinite(ret.total_time)) {
            throw std::runtime_error("FEL configuration contains a non-finite physical value");
        }

        ret.length_scale_in_jobfile   = lmult;
        ret.temporal_scale_in_jobfile = tmult;
        ret.charge                    = fel_config_detail::requiredNumber(root, "bunch/charge")
                                        * electron_charge_in_unit_charges;
        ret.mass =
            fel_config_detail::requiredNumber(root, "bunch/mass") * electron_mass_in_unit_masses;
        ret.num_particles = fel_config_detail::requiredInteger<uint64_t>(
            root, "bunch/number-of-particles");
        ret.mean_position = fel_config_detail::getVector<config::scalar, 3>(root, "bunch/position")
                            * lmult / unit_length_in_meters;
        ret.sigma_position =
            fel_config_detail::getVector<config::scalar, 3>(root, "bunch/sigma-position") * lmult
            / unit_length_in_meters;
        ret.position_truncations =
            fel_config_detail::getVector<config::scalar, 3>(root, "bunch/distribution-truncations")
            * lmult / unit_length_in_meters;
        ret.sigma_momentum =
            fel_config_detail::getVector<config::scalar, 3>(root, "bunch/sigma-momentum");

        ret.output_path = "../data/";
        if (root.has_path("output/path")) {
            ret.output_path = fel_config_detail::requiredString(root, "output/path");
            if (!ret.output_path.ends_with('/')) {
                ret.output_path.push_back('/');
            }
        }

        if (root.has_path("experimentation")) {
            const auto experimentation = root["experimentation"];
            if (!experimentation.dtype().is_object()) {
                throw std::runtime_error("Configuration value 'experimentation' must be an object");
            }
            for (conduit_index_t i = 0; i < experimentation.number_of_children(); ++i) {
                const auto option = experimentation.child(i);
                if (!option.dtype().is_number() || option.number_of_elements() != 1) {
                    throw std::runtime_error("Experimentation option '" + option.name()
                                             + "' must be a numeric scalar");
                }
                ret.experiment_options[option.name()] = option.to_double();
            }
        }

        return ret;
    } catch (const std::exception& error) {
        throw std::runtime_error("Failed to read FEL configuration '" + std::string(filepath)
                                 + "': " + error.what());
    }
}

#endif
