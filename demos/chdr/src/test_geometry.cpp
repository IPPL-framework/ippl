/**
 * @file test_geometry.cpp
 * @brief Unit-level regression checks for geometry predicates and YAML input.
 * @ingroup chdr_tests
 *
 * The small, hand-classified shapes exercise analytic volumes, closed boundaries,
 * invalid inputs and millimetre-to-metre conversion without constructing an IPPL
 * mesh. Kokkos reductions also call the predicates in the configured default
 * execution space: an OpenMP build checks CPU execution; GPU execution requires
 * a GPU-enabled build and a run on that backend.
 *
 * From the repository root after building test-chdr-geometry, run
 * @code{.sh}
 * ctest --test-dir build_openmp -R '^chdr\.mesh\.geometry$' --output-on-failure
 * @endcode
 * The separate mesh-1 integration runs and verify_mesh_output.py check distributed
 * material fields. This executable does not test MPI decomposition, electromagnetic
 * evolution, radiation or rendered figures.
 */
#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "BrickGeometry.h"
#include "ChdrMeshConfig.h"
#include "PrismGeometry.h"

namespace {
    /** @brief Fail the current check with a message propagated to main(). */
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    /**
     * @brief Check triangular-prism volume, boundaries, winding and rigid motion.
     *
     * Simple metre coordinates give a 6 m^3 reference volume. Known interior and
     * exterior points are checked on the host; two classifications are also
     * reduced in Kokkos's configured execution space. This is a focused predicate
     * test, not exhaustive coverage of all prism orientations or floating-point
     * distances near a face.
     * @throws std::runtime_error If a geometric expectation is not met.
     */
    void geometryChecks() {
        using Vertices = std::array<std::array<double, 3>, 3>;
        const Vertices vertices{{{0, 0, 0}, {2, 0, 0}, {0, 0, 2}}};
        const std::array<double, 3> axis{0, 1, 0};
        chdr::PrismGeometry prism(vertices, axis, 3);
        static_assert(std::is_trivially_copyable_v<chdr::PrismGeometry>);
        require(std::abs(prism.volume() - 6) < 1e-13, "triangle area times extrusion");
        require(prism.contains(.5, 1.5, .5), "interior point");
        require(prism.contains(0, 0, 0), "closed bottom vertex");
        require(prism.contains(1, 3, 1), "closed top/slanted boundary");
        require(!prism.contains(1.1, 1, 1.1), "point beyond sloping face");
        require(!prism.contains(.5, -.01, .5), "below bottom");
        require(!prism.contains(.5, 3.01, .5), "above top");
        require(!prism.contains(-.01, 1, .5), "outside triangle");
        auto reversedVertices = vertices;
        std::swap(reversedVertices[1], reversedVertices[2]);
        chdr::PrismGeometry reversed(reversedVertices, axis, 3);
        require(reversed.contains(.5, 1.5, .5) && !reversed.contains(1.1, 1, 1.1),
                "reversing vertex winding preserves geometry");
        auto invalidGeometry = [](const Vertices& points, const std::array<double, 3>& direction) {
            bool rejected = false;
            try {
                chdr::PrismGeometry invalid(points, direction, 3);
            } catch (const std::exception&) {
                rejected = true;
            }
            require(rejected, "invalid prism must be rejected");
        };
        invalidGeometry(vertices, {0, 2, 0});
        invalidGeometry(Vertices{{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}}, axis);
        invalidGeometry(vertices, {1, 0, 0});

        // An independent 90-degree rotation/translation checks arbitrary extrusion.
        Vertices moved{};
        for (int v = 0; v < 3; ++v)
            moved[v] = {10 - vertices[v][1], -4 + vertices[v][0], 7 + vertices[v][2]};
        chdr::PrismGeometry rotated(moved, std::array<double, 3>{-1, 0, 0}, 3);
        require(rotated.contains(8.5, -3.5, 7.5), "rotated interior point");
        require(!rotated.contains(8.5, -2.9, 8.1), "rotated exterior point");
        require(std::abs(rotated.volume() - prism.volume()) < 1e-13, "rotation preserves volume");

        int count = 0;
        Kokkos::parallel_reduce(
            "device prism point check", 2,
            KOKKOS_LAMBDA(int i, int& sum) {
                if (prism.contains(i == 0 ? .5 : 1.5, 1, .75))
                    ++sum;
            },
            count);
        require(count == 1, "device inside predicate agrees with known classification");
    }

    /**
     * @brief Check an axis-aligned brick, including all six closed faces.
     *
     * The reference volume is 24 m^3. Host checks cover corners, edges, exterior
     * points and invalid or unrepresentable dimensions. A Kokkos reduction checks
     * each face and one outward displacement per face on the configured backend.
     * @throws std::runtime_error If classification, volume or rejection fails.
     */
    void brickChecks() {
        const std::array<double, 3> lower{-2, -1, 4};
        const std::array<double, 3> size{2, 3, 4};
        const chdr::BrickGeometry brick(lower, size);
        static_assert(std::is_trivially_copyable_v<chdr::BrickGeometry>);
        require(std::abs(brick.volume() - 24) < 1e-13, "brick analytic volume");
        require(brick.contains(-1, .5, 6), "brick interior point");
        require(brick.contains(-2, -1, 4), "closed lower corner");
        require(brick.contains(0, 2, 8), "closed upper corner");
        require(brick.contains(-2, 2, 6), "closed brick edge");
        for (int d = 0; d < 3; ++d) {
            std::array<double, 3> point{-1, .5, 6};
            point[d] = lower[d];
            require(brick.contains(point[0], point[1], point[2]), "closed lower face");
            point[d] -= .01;
            require(!brick.contains(point[0], point[1], point[2]), "outside lower face");
            point[d] = lower[d] + size[d];
            require(brick.contains(point[0], point[1], point[2]), "closed upper face");
            point[d] += .01;
            require(!brick.contains(point[0], point[1], point[2]), "outside upper face");
        }
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        require(!brick.contains(nan, .5, 6), "NaN is not a point inside the brick");
        require(!brick.contains(-1, inf, 6), "infinity is outside the brick");
        auto invalidGeometry = [](const std::array<double, 3>& origin,
                                  const std::array<double, 3>& extent) {
            bool rejected = false;
            try {
                chdr::BrickGeometry invalid(origin, extent);
            } catch (const std::exception&) {
                rejected = true;
            }
            require(rejected, "invalid brick must be rejected");
        };
        invalidGeometry(lower, {0, 3, 4});
        invalidGeometry(lower, {2, -3, 4});
        invalidGeometry({nan, -1, 4}, size);
        invalidGeometry(lower, {2, inf, 4});
        invalidGeometry({1e300, 0, 0}, {1, 1, 1});             // unrepresentable upper face
        invalidGeometry({1e308, 0, 0}, {1e308, 1, 1});         // upper face overflow
        invalidGeometry({0, 0, 0}, {1e200, 1e200, 1e200});     // volume overflow
        invalidGeometry({0, 0, 0}, {1e-200, 1e-200, 1e-200});  // volume underflow

        // Every lower and upper face is included; displace that face outward for
        // the other six cases. Check all twelve on the configured Kokkos backend.
        int mismatches = 0;
        Kokkos::parallel_reduce(
            "device brick face check", 12,
            KOKKOS_LAMBDA(int i, int& count) {
                double point[3]           = {-1, .5, 6};
                const double lowerFace[3] = {-2, -1, 4};
                const double upperFace[3] = {0, 2, 8};
                const int face            = i % 6;
                const int dimension       = face / 2;
                point[dimension] = face % 2 == 0 ? lowerFace[dimension] : upperFace[dimension];
                if (i >= 6)
                    point[dimension] += face % 2 == 0 ? -.01 : .01;
                if (brick.contains(point[0], point[1], point[2]) != (i < 6))
                    ++count;
            },
            mismatches);
        require(mismatches == 0, "device brick classification includes faces, excludes exterior");
    }

    /**
     * @brief Check example YAML parsing, SI conversion and invalid-input rejection.
     *
     * Reads config.yaml beside this source file, then writes modified prism and
     * brick fixtures to a temporary file removed on scope exit. Tests selected
     * missing/unknown keys, types, units, extents, material values and preview
     * limits. It does not allocate a mesh or validate every possible YAML input.
     * @throws std::runtime_error If a parsed value or rejection is unexpected.
     */
    void configChecks() {
        const auto source = std::filesystem::path(__FILE__).parent_path() / "config.yaml";
        const auto cfg    = chdr::readMeshConfig(source.string());
        require(cfg.geometryType == chdr::GeometryType::Prism, "existing prism schema preserved");
        require(cfg.cells == std::array<int, 3>{40, 40, 64}, "quick example cell counts");
        require(std::abs(cfg.size[0] - .1) < 1e-15, "millimetres converted to metres");
        require(std::abs(cfg.lower[2] + .055) < 1e-15, "negative coordinates preserved");
        require(std::abs(chdr::PrismGeometry(cfg).volume() - 2.5e-5) < 1e-16,
                "configured physical volume");
        std::ifstream input(source);
        std::ostringstream buffer;
        buffer << input.rdbuf();
        const auto text = buffer.str();
        const auto temp =
            std::filesystem::temp_directory_path()
            / ("chdr-invalid-"
               + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
               + ".yaml");
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() {
                std::error_code e;
                std::filesystem::remove(path, e);
            }
        } cleanup{temp};
        auto rejected = [&](std::string invalid) {
            {
                std::ofstream output(temp);
                output << invalid;
            }
            bool failed = false;
            try {
                (void)chdr::readMeshConfig(temp.string());
            } catch (const std::exception&) {
                failed = true;
            }
            require(failed, "invalid configuration must be rejected");
        };
        rejected(text + "\nunrecognized_input: 1\n");
        rejected("schema_version: 1\nunits: {length: mm}\n");
        // Replacing just one required scalar tests unsupported physical units.
        auto invalidUnits = text;
        const auto unit   = invalidUnits.find("length: mm");
        require(unit != std::string::npos, "example has explicit mm units");
        invalidUnits.replace(unit, std::string("length: mm").size(), "length: parsec");
        rejected(invalidUnits);
        auto replaceAndReject = [&](const std::string& oldText, const std::string& newText) {
            auto changed     = text;
            const auto where = changed.find(oldText);
            require(where != std::string::npos, "validation fixture has expected input");
            changed.replace(where, oldText.size(), newText);
            rejected(changed);
        };
        replaceAndReject("cells: [40, 40, 64]", "cells: [40.5, 40, 64]");
        replaceAndReject("cells: [40, 40, 64]", "cells: [0, 40, 64]");
        replaceAndReject("cells: [40, 40, 64]", "cells: 40");
        replaceAndReject("height: 40.0", "height: 400.0");
        replaceAndReject("epsilon_r: 2.13", "epsilon_r: -2.13");
        replaceAndReject("preview_max_points_per_axis: 256",
                         "preview_max_points_per_axis: 1000000");

        const std::string brickGeometry =
            "geometry:\n"
            "  - name: radiator_brick\n"
            "    type: brick\n"
            "    material: radiator\n"
            "    lower_corner: [-25.0, -20.0, 0.0]\n"
            "    size: [25.0, 40.0, 50.0]\n\n";
        const auto geometryBegin = text.find("geometry:\n");
        const auto outputBegin   = text.find("output:\n", geometryBegin);
        require(geometryBegin != std::string::npos && outputBegin != std::string::npos,
                "fixture separates geometry and output mappings");
        auto brickText = text;
        brickText.replace(geometryBegin, outputBegin - geometryBegin, brickGeometry);
        {
            std::ofstream output(temp);
            output << brickText;
        }
        const auto brickCfg = chdr::readMeshConfig(temp.string());
        require(brickCfg.geometryType == chdr::GeometryType::Brick, "brick schema selected");
        require(std::abs(brickCfg.brickLower[0] + .025) < 1e-15
                    && std::abs(brickCfg.brickLower[1] + .020) < 1e-15
                    && brickCfg.brickLower[2] == 0.0,
                "brick lower corner converted from mm to metres");
        require(std::abs(brickCfg.brickSize[0] - .025) < 1e-15
                    && std::abs(brickCfg.brickSize[1] - .040) < 1e-15
                    && std::abs(brickCfg.brickSize[2] - .050) < 1e-15,
                "all brick dimensions converted from mm to metres");
        require(std::abs(chdr::BrickGeometry(brickCfg).volume() - 5e-5) < 1e-16,
                "configured brick physical volume");
        auto replaceBrickAndReject = [&](const std::string& oldText, const std::string& newText) {
            auto changed     = brickText;
            const auto where = changed.find(oldText);
            require(where != std::string::npos, "brick validation fixture has expected input");
            changed.replace(where, oldText.size(), newText);
            rejected(changed);
        };
        replaceBrickAndReject("type: brick", "type: sphere");
        replaceBrickAndReject("type: brick", "type: prism");  // brick-only keys in prism
        replaceBrickAndReject("type: brick", "type: brick\n    height: 40.0");
        replaceBrickAndReject("size: [25.0, 40.0, 50.0]", "size: [25.0, 0.0, 50.0]");
        replaceBrickAndReject("size: [25.0, 40.0, 50.0]", "size: [25.0, -40.0, 50.0]");
        replaceBrickAndReject("size: [25.0, 40.0, 50.0]", "size: [25.0, 400.0, 50.0]");
        replaceBrickAndReject("lower_corner: [-25.0, -20.0, 0.0]",
                              "lower_corner: [-250.0, -20.0, 0.0]");
        replaceBrickAndReject("    size: [25.0, 40.0, 50.0]\n", "");
        replaceBrickAndReject("    size: [25.0, 40.0, 50.0]\n", "    size: 25.0\n");
        replaceAndReject("type: prism", "type: brick");  // prism-only keys in brick
    }
}  // namespace

/**
 * @brief Initialize Kokkos and run the geometry/configuration checks in one process.
 * @ingroup chdr_tests
 * @param argc Number of command-line arguments passed to Kokkos initialization.
 * @param argv Command-line arguments; this test defines no additional options.
 * @return Zero when all checks pass; one for a caught check/configuration exception.
 *
 * A caught failure is printed to standard error, and Kokkos is finalized before
 * returning. Initialization errors occur before the check exception handler.
 */
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int status = 0;
    try {
        geometryChecks();
        brickChecks();
        configChecks();
        std::cout << "ChDR geometry/configuration checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    Kokkos::finalize();
    return status;
}
