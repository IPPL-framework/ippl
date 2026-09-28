/** @file Config.h
 * @brief Read the FEL mini-app's MITHRA-style JSON input and convert physical units.
 * @ingroup fel_config
 *
 * read_config() loads JSON through Catalyst Conduit. Length and time values use the
 * job file's mesh scales and are converted to the internal units in units.h.
 * The parser checks required values, numeric shapes and selected physical values;
 * it is not a complete physical-validity check. See read_config() for the schema,
 * defaults and the checks that are actually performed.
 */
#ifndef IPPL_FEL_CONFIG_H
#define IPPL_FEL_CONFIG_H

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

/** @addtogroup fel_config
 * @{ */

/** @brief Host-side FEL input after conversion to internal simulation units.
 *
 * read_config() value-initializes this structure and fills required values; plain
 * default construction does not initialize every scalar member. All lengths and
 * times below are in the internal units, except the two explicitly named job-file
 * conversion factors. The manager later converts longitudinal extents and run time
 * to its moving frame. No dielectric material specification is present here.
 */
struct config {
    using scalar = double;  ///< Floating-point type used for parsed physical values.

    // GRID PARAMETERS
    ippl::Vector<uint32_t, 3> resolution;  ///< Global cell counts (x,y,z); the parser permits zero.
    ippl::Vector<scalar, 3>
        extents;            ///< Lab-input domain lengths (x,y,z), in internal length units.
    scalar total_time;      ///< Lab-input run duration, in internal time units; manager divides by
                            ///< frame gamma.
    scalar timestep_ratio;  ///< Parsed top-level timestep-ratio, default 1; currently unused by the
                            ///< time-step selection.

    scalar length_scale_in_jobfile;    ///< SI metres per length unit in the input job file.
    scalar temporal_scale_in_jobfile;  ///< SI seconds per time unit in the input job file.

    // PARTICLE PARAMETERS
    scalar charge;  ///< Total bunch charge in internal units; input is a signed multiple of the
                    ///< positive elementary-charge magnitude.
    scalar mass;    ///< Total bunch mass in internal mass units; input is in electron-mass units.
    uint64_t num_particles;  ///< Requested particle count; generator's actual count can differ.
    bool space_charge;  ///< Enables deposition of rho in source component 0; current deposition
                        ///< remains enabled.

    // BUNCH PARAMETERS
    ippl::Vector<scalar, 3> mean_position;   ///< Requested initial bunch centre, in internal
                                             ///< lengths; manager subsequently recentres it.
    ippl::Vector<scalar, 3> sigma_position;  ///< Input position widths (x,y,z), converted to
                                             ///< internal lengths for the bunch generator.
    ippl::Vector<scalar, 3>
        position_truncations;  ///< Absolute distribution cutoff lengths, not multiples of sigma;
                               ///< generator uses x and z entries.
    ippl::Vector<scalar, 3> sigma_momentum;  ///< Dimensionless spread of gamma*beta components; no
                                             ///< unit conversion is applied.
    scalar bunch_gamma;  ///< Initial laboratory Lorentz factor; values below one are rejected.

    // UNDULATOR PARAMETERS
    scalar undulator_K;       ///< Dimensionless static-undulator deflection parameter K.
    scalar undulator_period;  ///< Static-undulator period, in internal length units.
    scalar undulator_length;  ///< Static-undulator length, in internal length units.

    std::string output_path;  ///< Output directory, default ../data/; parser ensures a trailing
                              ///< slash for overrides.
    std::unordered_map<std::string, double>
        experiment_options;  ///< Numeric experimentation options; empty if absent, names not
                             ///< validated here.
};

/** @brief Host-only Conduit readers with path-specific error messages.
 * Floating-point readers do not generally enforce finiteness or positivity;
 * the calling read_config() performs only its explicitly documented extra checks.
 */
namespace fel_config_detail {

    /** @brief Fetch a required configuration node.
     * @param root Parent node.
     * @param path Slash-separated path relative to root.
     * @return The existing node at path.
     * @throws std::runtime_error If path is absent.
     */
    inline conduit_cpp::Node requiredNode(const conduit_cpp::Node& root, const std::string& path) {
        if (!root.has_path(path)) {
            throw std::runtime_error("Missing required configuration value '" + path + "'");
        }
        return root[path];
    }

    /** @brief Read one typed numeric element without first narrowing it to double.
     * @param node Numeric scalar or array.
     * @param index Zero-based element index.
     * @param path Configuration path used in diagnostics.
     * @return The value as long double; nonfinite values are not rejected here.
     * @throws std::runtime_error For a nonnumeric node, invalid index or unsupported type.
     */
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

    /** @brief Fetch a required numeric scalar, with no finiteness or sign check.
     * @param root Parent node.
     * @param path Required scalar path.
     * @return The numeric value converted to double.
     * @throws std::runtime_error If missing, nonnumeric or not a single element.
     */
    inline double requiredNumber(const conduit_cpp::Node& root, const std::string& path) {
        const auto node = requiredNode(root, path);
        if (node.number_of_elements() != 1) {
            throw std::runtime_error("Configuration value '" + path + "' must be a scalar");
        }
        return static_cast<double>(numericElement(node, 0, path));
    }

    /** @brief Fetch a required string without converting other scalar types.
     * @param root Parent node.
     * @param path Required string path.
     * @return The string; empty strings are permitted by this helper.
     * @throws std::runtime_error If the node is missing or not a string.
     */
    inline std::string requiredString(const conduit_cpp::Node& root, const std::string& path) {
        const auto node = requiredNode(root, path);
        if (!node.dtype().is_string()) {
            throw std::runtime_error("Configuration value '" + path + "' must be a string");
        }
        return node.as_string();
    }

    /** @brief Cast a numeric value, checking range and integrality for integer targets.
     * @tparam Scalar Target arithmetic type.
     * @param value Value before conversion.
     * @param path Configuration path used in diagnostics.
     * @return The static_cast result; floating-point targets receive no extra validation.
     * @throws std::runtime_error For an integer target and nonfinite, fractional or
     * out-of-range input.
     */
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

    /** @brief Fetch a required exact integer representable in the requested type.
     * @tparam Scalar Integral destination type, enforced at compile time.
     * @param root Parent node.
     * @param path Required scalar path.
     * @return The representable integer; zero is not rejected by this helper.
     * @throws std::runtime_error For missing, malformed, nonintegral or out-of-range input.
     */
    template <typename Scalar>
    Scalar requiredInteger(const conduit_cpp::Node& root, const std::string& path) {
        static_assert(std::is_integral_v<Scalar>);
        const auto node = requiredNode(root, path);
        if (node.number_of_elements() != 1) {
            throw std::runtime_error("Configuration value '" + path + "' must be a scalar");
        }
        return checkedNumericCast<Scalar>(numericElement(node, 0, path), path);
    }

    /** @brief Fetch a numeric vector or broadcast a scalar with a stderr warning.
     * @tparam Scalar Component type; integer targets use checkedNumericCast().
     * @tparam Dim Required number of components when input is not a scalar.
     * @param root Parent node.
     * @param path Required numeric-array or scalar path.
     * @return The component-wise converted vector, or the scalar repeated Dim times.
     * @throws std::runtime_error For missing/nonnumeric input, wrong array length or
     * an invalid integer conversion. Floating-point finiteness is not checked here.
     */
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

/** @brief Fixed-size string key paired with a value for compile-time helper use.
 * @tparam N String-array size including the terminating null character.
 * @tparam T Associated value type.
 * @note The present unit parsers use chash(); this key/value helper is retained
 * but is not used by their switches.
 */
template <size_t N, typename T>
struct DefaultedStringLiteral {
    /** @brief Copy a literal and its associated value.
     * @param str String-array key, including its terminator.
     * @param val Associated value.
     */
    constexpr DefaultedStringLiteral(const char (&str)[N], const T val)
        : value(val) {
        std::copy_n(str, N, key);
    }

    T value;      ///< Associated value.
    char key[N];  ///< Copied string key including its null terminator.
};
/** @brief Structural string-literal wrapper for the compile-time chash() overload.
 * @tparam N String-array size including the terminating null character.
 */
template <size_t N>
struct StringLiteral {
    /** @brief Copy the complete literal into structural storage.
     * @param str Input literal including its null terminator.
     */
    constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }

    char value[N];  ///< Literal bytes including the terminating null character.
    /** @brief Pair the literal with an integer value.
     * @param t Associated integer.
     * @return A DefaultedStringLiteral holding the copied key and t.
     */
    constexpr DefaultedStringLiteral<N, int> operator>>(int t) const noexcept {
        return DefaultedStringLiteral<N, int>(value, t);
    }
    /** @brief Report the literal length excluding its terminator.
     * @return N minus one.
     */
    constexpr size_t size() const noexcept { return N - 1; }
};
/** @brief Compute the compile-time djb2-style hash used in unit-name switches.
 * @tparam lit Literal to hash, excluding its terminating null character.
 * @return The size_t recurrence h = 33*h + character, starting with 5381.
 * @note Hashing does not provide collision checking or input validation.
 */
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
/** @brief Hash a null-terminated unit name at run time.
 * @param val Valid null-terminated character string; must not be a null pointer.
 * @return The same djb2-style hash as the compile-time overload for the same bytes.
 */
inline size_t chash(const char* val) {
    size_t hash = 5381;
    int c;

    while ((c = *val++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }

    return hash;
}
/** @brief Hash a string through its null-terminated character representation.
 * @param _val Input string; hashing stops at its first null character.
 * @return The same djb2-style hash as the character-pointer overload.
 */
inline size_t chash(const std::string& _val) {
    size_t hash     = 5381;
    const char* val = _val.c_str();
    int c;

    while ((c = *val++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }

    return hash;
}
/** @brief Normalize a unit name by lowercasing and removing one trailing s.
 * @param str Input string, copied before modification.
 * @return Normalized name; no whitespace trimming or other plural rules are applied.
 */
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
/** @brief Decode mesh/time-scale as SI seconds per input time unit.
 * @param root Configuration tree containing the required string mesh/time-scale.
 * @return Seconds per named unit. Supports picosecond, nanosecond, microsecond,
 * millisecond and second; natural, planck-time, plancktime and pt mean the scaled
 * internal unit_time_in_seconds.
 * @throws std::runtime_error If the scale node is missing or not a string.
 * @note Names are lowercased and a trailing s is removed. An unknown name emits a
 * stderr message and returns 1.0 (seconds); it does not reject the configuration.
 */
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
/** @brief Decode mesh/length-scale as SI metres per input length unit.
 * @param root Configuration tree containing the required string mesh/length-scale.
 * @return Metres per named unit. Supports picometer, nanometer, micrometer,
 * millimeter and meter; natural, planck-length, plancklength and pl mean the scaled
 * internal unit_length_in_meters.
 * @throws std::runtime_error If the scale node is missing or not a string.
 * @note Names are lowercased and a trailing s is removed. An unknown name emits a
 * stderr message and returns 1.0 (metres); it does not reject the configuration.
 */
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
/** @brief Read a JSON job file into the FEL's host-side configuration.
 * @param filepath Valid null-terminated path to the input JSON file.
 * @return Parsed configuration in the internal units defined in units.h.
 * @throws std::runtime_error For required-value/type/shape errors, failed integer
 * conversions, gamma below one, the selected nonfinite values listed below, or
 * an invalid experimentation object. Caught standard exceptions are rethrown with
 * the input filename; loading and syntax errors originate in Conduit.
 *
 * Required input paths and meanings are:
 * - mesh/length-scale, mesh/time-scale: strings understood by the scale helpers.
 * - mesh/extents, mesh/resolution: three lengths and three uint32_t cell counts.
 * - mesh/total-time, mesh/space-charge: duration and numeric zero/nonzero switch.
 * - bunch/charge, bunch/mass: total bunch charge in elementary-charge units and
 *   total bunch mass in electron-mass units. No electron-sign inversion is applied.
 * - bunch/number-of-particles, bunch/gamma: uint64_t count and laboratory Lorentz factor.
 * - bunch/position, bunch/sigma-position, bunch/distribution-truncations: three
 *   lengths each. Cutoffs are absolute lengths, not dimensionless sigma multiples.
 * - bunch/sigma-momentum: three dimensionless gamma*beta spreads.
 * - undulator/static-undulator/undulator-parameter, period, length: dimensionless
 *   K followed by two lengths in the selected job-file length unit.
 *
 * Optional timestep-ratio defaults to 1 but does not currently alter the solver
 * time step. Optional output/path defaults to ../data/; overrides gain a trailing
 * slash. Optional experimentation is a mapping of arbitrary names to numeric
 * scalars and otherwise remains empty. Unknown configuration keys are ignored.
 * Numeric vector fields accept either three components or scalar broadcast.
 *
 * Conversion follows @f$x_{\rm internal}=x_{\rm input}s_x/L_0@f$ and
 * @f$t_{\rm internal}=t_{\rm input}s_t/T_0@f$, with SI job-file scales
 * @f$s_x,s_t@f$ and internal base units @f$L_0,T_0@f$ from units.h.
 * Only extents, total time, undulator period and undulator length receive an
 * explicit finiteness check after conversion. Integer range checks permit zero;
 * most other values are not checked for positivity or finiteness. In particular,
 * the gamma comparison rejects values below one but not NaN. Successful parsing
 * therefore does not establish that the full simulation is physically valid.
 */
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
        ret.num_particles =
            fel_config_detail::requiredInteger<uint64_t>(root, "bunch/number-of-particles");
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

/** @} */

#endif
