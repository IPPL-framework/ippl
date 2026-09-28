/** @file ChdrMeshConfig.h
 * @brief Host-side YAML parsing, SI-unit conversion and analytic geometry validation.
 * @ingroup chdr_config
 *
 * The configuration describes one dielectric radiator in a uniform Cartesian domain.
 * It prepares geometry and allocation inputs; it does not define a Maxwell update,
 * a particle source or absorbing boundary conditions.
 */
#ifndef IPPL_CHDR_MESH_CONFIG_H
#define IPPL_CHDR_MESH_CONFIG_H

#include <algorithm>
#include <array>
#include <catalyst_conduit.hpp>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

/** @brief Geometry, input and output utilities for the ChDR mesh demonstrator. */
namespace chdr {

    /** @brief Supported analytic radiator shapes before cell-centre sampling.
     * @ingroup chdr_config
     */
    enum class GeometryType {
        Prism,  ///< A triangular face extruded along its perpendicular unit axis.
        Brick   ///< An axis-aligned rectangular volume.
    };

    /** @brief Validated input for the geometry-only ChDR mesh demonstrator.
     * @ingroup chdr_config
     *
     * All stored lengths are SI metres. For a prism, the three vertices define the
     * bottom triangular face; height extrudes that face along the unit axis. A brick
     * is axis-aligned and defined by its lower corner and positive edge lengths.
     * For dimension @f$d@f$ and zero-based global cell index @f$i_d@f$, the spacing
     * and material sampling position are
     * @f[
     * h_d = L_d/N_d,\qquad x_{i_d}=\ell_d+(i_d+1/2)h_d.
     * @f]
     * Here @f$\ell_d@f$, @f$L_d@f$ and @f$N_d@f$ are lower, size and cells.
     * The material of the cell centre fills the whole cell (a staircase interface);
     * no volume-fraction averaging is performed.
     *
     * @note Default construction alone does not produce a valid configuration:
     * domain and geometry dimensions still need to be populated. readMeshConfig()
     * validates the supplied YAML and preserves the optional-field defaults below.
     */
    struct MeshConfig {
        std::array<double, 3> lower{};  ///< Computational-domain lower corner (x,y,z), in metres.
        std::array<double, 3> size{};   ///< Positive domain lengths (x,y,z), in metres.
        std::array<int, 3> cells{};     ///< Positive global cell counts along x, y and z.
        std::array<bool, 3> decompose{false, false,
                                      true};  ///< Allowed MPI split axes; z only by default.
        double backgroundEpsilon =
            1.0;  ///< Positive dimensionless background relative permittivity.
        double radiatorEpsilon = 2.13;  ///< Positive dimensionless radiator relative permittivity.
        GeometryType geometryType =
            GeometryType::Prism;  ///< Selects the active shape fields; default prism.
        std::array<std::array<double, 3>, 3>
            vertices{};  ///< Prism bottom-face vertices, in metres; unused for bricks.
        std::array<double, 3>
            axis{};  ///< Dimensionless prism extrusion unit vector; unused for bricks.
        std::array<double, 3> brickLower{};  ///< Brick lower corner, in metres; unused for prisms.
        std::array<double, 3>
            brickSize{};      ///< Positive brick edge lengths, in metres; unused for prisms.
        double height = 0.0;  ///< Positive prism extrusion length, in metres; unused for bricks.
        std::string outputPath = "mesh-1-output";  ///< Output directory; relative paths use the
                                                   ///< process working directory.
        int previewMaxPointsPerAxis =
            256;  ///< Maximum sampled slice points per axis; default 256, allowed 2..1024.
        double inputLengthInMeters =
            1.0;  ///< Metres per input length unit; applied once during parsing.
        std::string inputLengthUnit = "m";  ///< Original YAML length-unit label: m, cm, mm or um.
    };

    /** @brief Host-only parsing and validation helpers shared by geometry constructors.
     * @ingroup chdr_config
     *
     * Call readMeshConfig() for application input. These helpers provide precise
     * configuration-path diagnostics and are also exposed to the small analytic
     * geometry constructors; none runs inside a Kokkos device kernel.
     */
    namespace mesh_config_detail {

        /** @brief Report a configuration error with its YAML path.
         * @param path Slash-separated path naming the offending input.
         * @param reason Explanation of the violated constraint.
         * @throws std::runtime_error Always; the message includes path and reason.
         */
        inline void fail(const std::string& path, const std::string& reason) {
            throw std::runtime_error("ChDR configuration '" + path + "': " + reason);
        }

        /** @brief Obtain a required child node without silently supplying a default.
         * @param root Mapping or parent node to inspect.
         * @param path Slash-separated child path relative to root.
         * @return The existing node at path.
         * @throws std::runtime_error If the child is missing.
         */
        inline conduit_cpp::Node required(const conduit_cpp::Node& root, const std::string& path) {
            if (!root.has_path(path)) {
                fail(path, "missing required value");
            }
            return root[path];
        }

        /** @brief Reject misspelled keys instead of silently running a different case.
         * @param node Mapping whose direct children are checked.
         * @param path Input path used in diagnostics.
         * @param allowed Complete set of permitted direct child names.
         * @throws std::runtime_error If node is not a mapping or contains an unknown key.
         */
        inline void keys(const conduit_cpp::Node& node, const std::string& path,
                         std::initializer_list<std::string> allowed) {
            if (!node.dtype().is_object()) {
                fail(path, "expected a mapping");
            }
            for (conduit_index_t i = 0; i < node.number_of_children(); ++i) {
                const auto name = node.child(i).name();
                if (std::find(allowed.begin(), allowed.end(), name) == allowed.end()) {
                    fail(path + "/" + name, "unsupported key");
                }
            }
        }

        /** @brief Read one supported Conduit numeric element without integer truncation.
         * @param node Typed numeric scalar or array.
         * @param index Zero-based element index.
         * @param path Input path used in diagnostics.
         * @return The element converted to long double; finiteness is checked by callers.
         * @throws std::runtime_error If the type or index is unsupported.
         */
        inline long double numericElement(const conduit_cpp::Node& node, conduit_index_t index,
                                          const std::string& path) {
            if (!node.dtype().is_number() || index < 0 || index >= node.number_of_elements()) {
                fail(path, "expected a numeric value");
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
            fail(path, "unsupported numeric representation");
            return 0.0L;
        }

        /** @brief Read exactly one finite numeric scalar as a double.
         * @param node Input scalar node; arrays are rejected.
         * @param path Input path used in diagnostics.
         * @return The finite scalar without unit conversion.
         * @throws std::runtime_error For a nonscalar, nonnumeric or nonfinite value.
         */
        inline double number(const conduit_cpp::Node& node, const std::string& path) {
            if (!node.dtype().is_number() || node.number_of_elements() != 1) {
                fail(path, "expected one numeric scalar");
            }
            const double result = static_cast<double>(numericElement(node, 0, path));
            if (!std::isfinite(result)) {
                fail(path, "value must be finite");
            }
            return result;
        }

        /** @brief Convert an exact, representable integer without rounding fractions.
         * @param value Numeric input before conversion.
         * @param path Input path used in diagnostics.
         * @return The value as an int.
         * @throws std::runtime_error For fractions, nonfinite values or int overflow.
         */
        inline int integer(long double value, const std::string& path) {
            if (!std::isfinite(value) || std::trunc(value) != value
                || value < std::numeric_limits<int>::lowest()
                || value > std::numeric_limits<int>::max()) {
                fail(path, "expected an in-range integer");
            }
            return static_cast<int>(value);
        }

        /** @brief Read a string without converting another YAML scalar type.
         * @param node Input node expected to hold a string.
         * @param path Input path used in diagnostics.
         * @return The string; this helper permits an empty string.
         * @throws std::runtime_error If the node is not a string.
         */
        inline std::string string(const conduit_cpp::Node& node, const std::string& path) {
            if (!node.dtype().is_string()) {
                fail(path, "expected a string");
            }
            return node.as_string();
        }

        /** @brief Read either a typed numeric array or a YAML list of scalar numbers.
         * Scalar broadcasting is deliberately unsupported: a domain vector has three
         * explicitly specified components, including when their values are equal.
         * @param node Three-element numeric array or list of numeric scalars.
         * @param path Input path used in diagnostics.
         * @return The three finite components, without unit conversion.
         * @throws std::runtime_error For another shape, nonnumeric or nonfinite values.
         */
        inline std::array<long double, 3> numericTriple(const conduit_cpp::Node& node,
                                                        const std::string& path) {
            std::array<long double, 3> result{};
            if (node.dtype().is_number() && node.number_of_elements() == 3) {
                for (int i = 0; i < 3; ++i) {
                    result[i] = numericElement(node, i, path);
                }
            } else if (node.dtype().is_list() && node.number_of_children() == 3) {
                for (int i = 0; i < 3; ++i) {
                    const auto child = node.child(i);
                    if (!child.dtype().is_number() || child.number_of_elements() != 1) {
                        fail(path, "expected exactly three numeric scalars");
                    }
                    result[i] = numericElement(child, 0, path);
                }
            } else {
                fail(path, "expected exactly three numeric scalars");
            }
            for (const auto value : result) {
                if (!std::isfinite(value)) {
                    fail(path, "components must be finite");
                }
            }
            return result;
        }

        /** @brief Read and scale three coordinates, dimensions or direction components.
         * @param node Three-component numeric input.
         * @param path Input path used in diagnostics.
         * @param factor Multiplicative unit conversion; defaults to one for unit vectors.
         * @return Three finite doubles after applying factor.
         * @throws std::runtime_error For invalid triples or overflow on conversion.
         */
        inline std::array<double, 3> vector(const conduit_cpp::Node& node, const std::string& path,
                                            double factor = 1.0) {
            const auto values = numericTriple(node, path);
            std::array<double, 3> result{};
            for (int i = 0; i < 3; ++i) {
                result[i] = static_cast<double>(values[i] * factor);
                if (!std::isfinite(result[i])) {
                    fail(path, "components overflow after unit conversion");
                }
            }
            return result;
        }

        /** @brief Read a Boolean despite Conduit's version-dependent YAML representation.
         * @param node String true/false or numeric scalar 1/0.
         * @param path Input path used in diagnostics.
         * @return The decoded Boolean value.
         * @throws std::runtime_error For any other value or node type.
         */
        inline bool boolean(const conduit_cpp::Node& node, const std::string& path) {
            // Conduit has no separate boolean dtype; YAML booleans may be strings or
            // numeric 0/1 depending on the installed Conduit version.
            if (node.dtype().is_string()) {
                const auto value = node.as_string();
                if (value == "true")
                    return true;
                if (value == "false")
                    return false;
            } else if (node.dtype().is_number() && node.number_of_elements() == 1) {
                const auto value = numericElement(node, 0, path);
                if (value == 0.0L)
                    return false;
                if (value == 1.0L)
                    return true;
            }
            fail(path, "expected true/false (or numeric 1/0)");
            return false;
        }

        /** @brief Read the enabled MPI decomposition axes, with at least one enabled.
         * @param node Three-element Boolean list or numeric 0/1 array, ordered x,y,z.
         * @param path Input path used in diagnostics.
         * @return The three decomposition flags.
         * @throws std::runtime_error For invalid entries, shape or three false flags.
         */
        inline std::array<bool, 3> booleanTriple(const conduit_cpp::Node& node,
                                                 const std::string& path) {
            std::array<bool, 3> result{};
            if (node.dtype().is_list() && node.number_of_children() == 3) {
                for (int i = 0; i < 3; ++i)
                    result[i] = boolean(node.child(i), path);
            } else if (node.dtype().is_number() && node.number_of_elements() == 3) {
                for (int i = 0; i < 3; ++i) {
                    const auto value = numericElement(node, i, path);
                    if (value != 0.0L && value != 1.0L)
                        fail(path, "components must be 0 or 1");
                    result[i] = value == 1.0L;
                }
            } else {
                fail(path, "expected exactly three booleans");
            }
            if (!result[0] && !result[1] && !result[2]) {
                fail(path, "at least one decomposition axis must be enabled");
            }
            return result;
        }

        /** @brief Compute a three-dimensional Euclidean scalar product on the host.
         * @param a First vector.
         * @param b Second vector.
         * @return @f$\sum_{d=0}^2 a_d b_d@f$, with the product of the input units.
         */
        inline double dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }

        /** @brief Compute a right-handed three-dimensional cross product on the host.
         * @param a First vector.
         * @param b Second vector.
         * @return @f$\mathbf a\times\mathbf b@f$, with the product of the input units.
         */
        inline std::array<double, 3> cross(const std::array<double, 3>& a,
                                           const std::array<double, 3>& b) {
            return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0]};
        }

        /** @brief Geometry validation shared by the YAML reader and host constructor.
         * @param vertices Three finite bottom-face coordinates, in metres.
         * @param axis Extrusion direction, whose dimensionless norm must be one within 1e-12.
         * @param height Positive finite extrusion length, in metres.
         * @return Length tolerance @f$\tau=64\epsilon_{\rm mach}s@f$ in metres, where
         * @f$s@f$ is the maximum of height, absolute vertex components and the lengths
         * of the two edges from vertex zero.
         * @throws std::runtime_error For nonfinite data, a nonunit axis, nonpositive
         * height, a degenerate triangle, nonfinite/nonpositive volume or a face not
         * perpendicular to axis within the length tolerance.
         *
         * Geometry predicates assign points within @f$\tau@f$ of a face to the
         * radiator. This absorbs floating-point roundoff; it is not a physical gap,
         * a mesh spacing or a volume-fraction smoothing width. It does not depend
         * on MPI layout or resolution. Domain containment is checked separately by
         * readMeshConfig(). This helper does not normalize axis or modify vertices.
         */
        inline double validatePrism(const std::array<std::array<double, 3>, 3>& vertices,
                                    const std::array<double, 3>& axis, double height) {
            if (!std::isfinite(height) || height <= 0.0) {
                fail("geometry/height", "must be finite and positive");
            }
            double scale = height;
            for (const auto& vertex : vertices) {
                for (const auto value : vertex) {
                    if (!std::isfinite(value))
                        fail("geometry/vertices", "must be finite");
                    scale = std::max(scale, std::abs(value));
                }
            }
            for (const auto value : axis) {
                if (!std::isfinite(value))
                    fail("geometry/axis", "must be finite");
            }
            const double axisNorm = std::sqrt(dot(axis, axis));
            if (!std::isfinite(axisNorm) || std::abs(axisNorm - 1.0) > 1.0e-12) {
                fail("geometry/axis", "must be a unit vector");
            }
            std::array<double, 3> edge0{}, edge1{};
            for (int d = 0; d < 3; ++d) {
                edge0[d] = vertices[1][d] - vertices[0][d];
                edge1[d] = vertices[2][d] - vertices[0][d];
            }
            const double length0   = std::sqrt(dot(edge0, edge0));
            const double length1   = std::sqrt(dot(edge1, edge1));
            scale                  = std::max({scale, length0, length1});
            const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
            const auto normal      = cross(edge0, edge1);
            const double twiceArea = std::sqrt(dot(normal, normal));
            if (!std::isfinite(twiceArea) || !std::isfinite(scale)
                || twiceArea <= 64.0 * std::numeric_limits<double>::epsilon() * length0 * length1
                || !std::isfinite(0.5 * twiceArea * height) || 0.5 * twiceArea * height <= 0.0) {
                fail("geometry/vertices",
                     "triangle must have a finite, nonzero area and prism volume");
            }
            if (std::abs(dot(edge0, axis)) > tolerance || std::abs(dot(edge1, axis)) > tolerance) {
                fail("geometry/vertices",
                     "bottom face must be perpendicular to the extrusion axis");
            }
            return tolerance;
        }

        /** @brief Validate a finite axis-aligned brick with representable faces and volume.
         * @param lower Lower corner (x,y,z), in metres.
         * @param size Positive edge lengths (x,y,z), in metres.
         * @return @f$\tau=64\epsilon_{\rm mach}s@f$ in metres, with @f$s@f$ the largest
         * edge length or absolute coordinate of either lower or upper corner.
         * @throws std::runtime_error For nonfinite coordinates, nonpositive lengths,
         * upper faces not representable beyond their lower faces, or a volume that
         * is not positive and finite when stored as a double.
         *
         * The tolerance has the same boundary-inclusive meaning as validatePrism();
         * it is independent of mesh resolution and decomposition. Domain containment
         * is checked separately by readMeshConfig(). Inputs are not modified.
         */
        inline double validateBrick(const std::array<double, 3>& lower,
                                    const std::array<double, 3>& size) {
            double scale       = 0.0;
            long double volume = 1.0L;
            for (int d = 0; d < 3; ++d) {
                if (!std::isfinite(lower[d]))
                    fail("geometry/lower_corner", "components must be finite");
                const double upper = lower[d] + size[d];
                if (!std::isfinite(size[d]) || size[d] <= 0.0 || !std::isfinite(upper)
                    || upper <= lower[d]) {
                    fail("geometry/size",
                         "components must be finite, positive and representable at the origin");
                }
                scale = std::max({scale, std::abs(lower[d]), std::abs(upper), size[d]});
                volume *= static_cast<long double>(size[d]);
            }
            const double storedVolume = static_cast<double>(volume);
            if (!std::isfinite(storedVolume) || storedVolume <= 0.0)
                fail("geometry/size", "brick must have finite, positive representable volume");
            return 64.0 * std::numeric_limits<double>::epsilon() * scale;
        }

    }  // namespace mesh_config_detail

    /** @brief Read the version-1 YAML geometry schema using FEL's Conduit dependency.
     * @ingroup chdr_config
     * @param path Path of a YAML configuration file, relative to the working directory
     * or absolute. The file must be readable by the calling process.
     * @return A validated MeshConfig with all lengths converted to metres and a
     * normalized prism axis; shape-inactive fields retain their defaults.
     * @throws std::runtime_error For unknown keys, malformed shapes, unsupported
     * materials/geometries, nonphysical dimensions, or a radiator outside the domain.
     *
     * The schema accepts exactly one prism or brick and two distinct named materials.
     * Both materials require finite positive relative permittivity and nonmagnetic
     * relative permeability equal to one. Dimensions use m, cm, mm or um; cell counts
     * are positive integers and centres must be representable at the domain origin.
     * The entire radiator must fit inside the domain within the geometric roundoff
     * tolerance. Only cell_center_staircase material sampling is supported.
     *
     * Optional output and decomposition entries use the defaults in MeshConfig.
     * Unknown and wrong-shape keys are rejected instead of silently ignored.
     * Parsing and validation happen on the host, before distributed field allocation;
     * this function itself neither communicates with MPI nor allocates device fields.
     * Errors while loading or parsing the file are handled by the Conduit dependency.
     */
    inline MeshConfig readMeshConfig(const std::string& path) {
        namespace detail = mesh_config_detail;
        conduit_cpp::Node root;
        conduit_node_load(conduit_cpp::c_node(&root), path.c_str(), "yaml");
        detail::keys(
            root, "root",
            {"schema_version", "units", "domain", "mesh", "materials", "geometry", "output"});
        const auto schema = detail::required(root, "schema_version");
        if (detail::number(schema, "schema_version") != 1.0) {
            detail::fail("schema_version", "only version 1 is supported");
        }

        MeshConfig config;
        const auto units = detail::required(root, "units");
        detail::keys(units, "units", {"length"});
        config.inputLengthUnit = detail::string(detail::required(units, "length"), "units/length");
        if (config.inputLengthUnit == "m")
            config.inputLengthInMeters = 1.0;
        else if (config.inputLengthUnit == "cm")
            config.inputLengthInMeters = 1.0e-2;
        else if (config.inputLengthUnit == "mm")
            config.inputLengthInMeters = 1.0e-3;
        else if (config.inputLengthUnit == "um")
            config.inputLengthInMeters = 1.0e-6;
        else
            detail::fail("units/length", "supported units are m, cm, mm, um");

        const auto domain = detail::required(root, "domain");
        detail::keys(domain, "domain", {"lower_corner", "size", "background"});
        config.lower = detail::vector(detail::required(domain, "lower_corner"),
                                      "domain/lower_corner", config.inputLengthInMeters);
        config.size  = detail::vector(detail::required(domain, "size"), "domain/size",
                                      config.inputLengthInMeters);
        const auto background =
            detail::string(detail::required(domain, "background"), "domain/background");
        for (int d = 0; d < 3; ++d) {
            if (config.size[d] <= 0.0 || !std::isfinite(config.lower[d] + config.size[d])
                || config.lower[d] + config.size[d] <= config.lower[d]) {
                detail::fail("domain/size",
                             "components must be positive and representable at the origin");
            }
        }

        const auto mesh = detail::required(root, "mesh");
        detail::keys(mesh, "mesh", {"cells", "material_sampling", "decompose"});
        const auto cells = detail::numericTriple(detail::required(mesh, "cells"), "mesh/cells");
        for (int d = 0; d < 3; ++d) {
            config.cells[d] = detail::integer(cells[d], "mesh/cells");
            if (config.cells[d] <= 0)
                detail::fail("mesh/cells", "components must be positive");
            const auto spacing = config.size[d] / config.cells[d];
            if (!(spacing > 0.0) || config.lower[d] + 0.5 * spacing == config.lower[d]) {
                detail::fail("mesh/cells",
                             "cell centres are not representable at this coordinate scale");
            }
        }
        if (detail::string(detail::required(mesh, "material_sampling"), "mesh/material_sampling")
            != "cell_center_staircase") {
            detail::fail("mesh/material_sampling", "only cell_center_staircase is supported");
        }
        if (mesh.has_path("decompose")) {
            config.decompose = detail::booleanTriple(mesh["decompose"], "mesh/decompose");
        }

        const auto geometry = detail::required(root, "geometry");
        if (!geometry.dtype().is_list() || geometry.number_of_children() != 1) {
            detail::fail("geometry", "expected a list containing exactly one prism or brick");
        }
        const auto shape = geometry.child(0);
        const auto type  = detail::string(detail::required(shape, "type"), "geometry/0/type");
        if (type == "prism") {
            config.geometryType = GeometryType::Prism;
            detail::keys(shape, "geometry/0",
                         {"name", "type", "material", "vertices", "axis", "height"});
        } else if (type == "brick") {
            config.geometryType = GeometryType::Brick;
            detail::keys(shape, "geometry/0", {"name", "type", "material", "lower_corner", "size"});
        } else {
            detail::fail("geometry/0/type", "only prism or brick is supported");
        }
        if (detail::string(detail::required(shape, "name"), "geometry/0/name").empty()) {
            detail::fail("geometry/0/name", "must not be empty");
        }
        const auto radiator =
            detail::string(detail::required(shape, "material"), "geometry/0/material");
        if (background.empty() || radiator.empty() || background == radiator
            || background.find('/') != std::string::npos
            || radiator.find('/') != std::string::npos) {
            detail::fail("materials",
                         "background and radiator must reference distinct, nonempty material names "
                         "without '/'");
        }
        const auto materials = detail::required(root, "materials");
        detail::keys(materials, "materials", {background, radiator});
        const auto readMaterial = [&materials](const std::string& name) {
            const auto material = detail::required(materials, name);
            detail::keys(material, "materials/" + name, {"epsilon_r", "mu_r"});
            const double epsilon = detail::number(detail::required(material, "epsilon_r"),
                                                  "materials/" + name + "/epsilon_r");
            const double mu =
                detail::number(detail::required(material, "mu_r"), "materials/" + name + "/mu_r");
            if (epsilon <= 0.0)
                detail::fail("materials/" + name + "/epsilon_r", "must be positive");
            if (mu != 1.0)
                detail::fail("materials/" + name + "/mu_r",
                             "only nonmagnetic mu_r = 1 is supported");
            return epsilon;
        };
        config.backgroundEpsilon = readMaterial(background);
        config.radiatorEpsilon   = readMaterial(radiator);
        if (config.geometryType == GeometryType::Prism) {
            const auto vertices = detail::required(shape, "vertices");
            if (!vertices.dtype().is_list() || vertices.number_of_children() != 3) {
                detail::fail("geometry/0/vertices", "expected exactly three 3D vertices");
            }
            for (int i = 0; i < 3; ++i) {
                config.vertices[i] =
                    detail::vector(vertices.child(i), "geometry/0/vertices/" + std::to_string(i),
                                   config.inputLengthInMeters);
            }
            config.axis   = detail::vector(detail::required(shape, "axis"), "geometry/0/axis");
            config.height = detail::number(detail::required(shape, "height"), "geometry/0/height")
                            * config.inputLengthInMeters;
            const auto tolerance =
                detail::validatePrism(config.vertices, config.axis, config.height);
            // Remove only floating-point roundoff from an already validated unit axis.
            const auto axisNorm = std::sqrt(detail::dot(config.axis, config.axis));
            for (auto& component : config.axis)
                component /= axisNorm;
            for (const auto& vertex : config.vertices) {
                for (int d = 0; d < 3; ++d) {
                    const double top   = vertex[d] + config.height * config.axis[d];
                    const double upper = config.lower[d] + config.size[d];
                    if (!std::isfinite(top) || vertex[d] < config.lower[d] - tolerance
                        || vertex[d] > upper + tolerance || top < config.lower[d] - tolerance
                        || top > upper + tolerance) {
                        detail::fail("geometry/0",
                                     "entire prism must lie inside the computational domain");
                    }
                }
            }
        } else {
            config.brickLower =
                detail::vector(detail::required(shape, "lower_corner"), "geometry/0/lower_corner",
                               config.inputLengthInMeters);
            config.brickSize = detail::vector(detail::required(shape, "size"), "geometry/0/size",
                                              config.inputLengthInMeters);
            const auto tolerance = detail::validateBrick(config.brickLower, config.brickSize);
            for (int d = 0; d < 3; ++d) {
                const double top   = config.brickLower[d] + config.brickSize[d];
                const double upper = config.lower[d] + config.size[d];
                if (config.brickLower[d] < config.lower[d] - tolerance || top > upper + tolerance) {
                    detail::fail("geometry/0",
                                 "entire brick must lie inside the computational domain");
                }
            }
        }
        if (root.has_path("output")) {
            const auto output = root["output"];
            detail::keys(output, "output", {"path", "preview_max_points_per_axis"});
            if (output.has_path("path")) {
                config.outputPath = detail::string(output["path"], "output/path");
                if (config.outputPath.empty())
                    detail::fail("output/path", "must not be empty");
            }
            if (output.has_path("preview_max_points_per_axis")) {
                const auto value = output["preview_max_points_per_axis"];
                config.previewMaxPointsPerAxis =
                    detail::integer(detail::number(value, "output/preview_max_points_per_axis"),
                                    "output/preview_max_points_per_axis");
                if (config.previewMaxPointsPerAxis < 2 || config.previewMaxPointsPerAxis > 1024) {
                    detail::fail("output/preview_max_points_per_axis",
                                 "must be between 2 and 1024");
                }
            }
        }
        return config;
    }

}  // namespace chdr

#endif
