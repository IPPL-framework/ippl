/**
 * @file TestCosmologyICRandom.cpp
 * @brief Independent byte-oracle and host/device checks for the optional physical-mode RNG.
 * @ingroup cosmology_diagnostics
 * @see cosmology_validation cosmology_numerics cosmology::modeGaussian
 *
 * Golden digests, binary64 uniforms and Gaussian values were generated on Merlin
 * from frozen python/gaussian_fixture.py (SHA256
 * 3be089f091e971cb240e93411b9d1c5f2d0756928c768ef8d8025cbbad2ea24a)
 * with Python hashlib/struct/math, independently of this C++ SHA256 code.
 * Integer digests and uniforms must match exactly; Gaussian values allow only
 * 8e-15 absolute error for host/device log/sqrt/trigonometric roundoff. No
 * spectrum normalization, fitted amplitude or statistical acceptance enters
 * these fixed-vector and conjugation tests.
 */
#include "CosmologyICRandom.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

/** @brief Seed and signed physical integer mode transferred to execution memory. */
struct ModeInput {
    std::uint64_t seed_m; ///< Full seed, including high unsigned bits.
    std::int32_t x_m; ///< Signed physical x mode; no grid/storage-index encoding.
    std::int32_t y_m; ///< Signed physical y mode; independent of the sampling grid.
    std::int32_t z_m; ///< Signed physical z mode; independent of MPI ownership.
};

/** @brief Independent expected digest, uniforms and Gaussian value for one mode. */
struct Oracle {
    ModeInput input_m; ///< Requested seed and signed mode.
    std::uint32_t words_m[8]; ///< Standard big-endian SHA256 words from Python hashlib.
    double u1_m; ///< First exactly representable open uniform, written as a hexadecimal literal.
    double u2_m; ///< Second exactly representable open uniform, written as a hexadecimal literal.
    double real_m; ///< Real Python math Box-Muller component, before P/V scaling.
    double imaginary_m; ///< Imaginary Python math Box-Muller component, before P/V scaling.
};

constexpr double GaussianTolerance = 8e-15; ///< Absolute device-transcendental roundoff allowance.
/** @brief Frozen independent Python SHA256/Box-Muller vectors computed on Merlin. */
constexpr std::array<Oracle, 9> Oracles{{
    {{UINT64_C(0), 1, 0, 0},
     {0xe82ce7ecU, 0x12352b95U, 0xb5ef008cU, 0x9397ae47U,
      0x4ab8649cU, 0x12064f80U, 0x4e158831U, 0x98caf15eU},
     0x1.2a566a25d9ce5p-1, 0x1.1eba5e4e3003ap-2,
     -0.13774465124378754, 0.7218901913834072},
    {{UINT64_C(17), 2, -1, 0},
     {0x3aa1fc8cU, 0xcc68839eU, 0x6d69fa49U, 0xb9cd5d1cU,
      0x67a6ac4eU, 0x066a60cbU, 0x3da7e149U, 0xa6307173U},
     0x1.3d06d19919f95p-1, 0x1.c5dcdb949fa68p-4,
     0.531216766543139, 0.44401303731530245},
    {{UINT64_C(17), -2, 1, 0},
     {0x3aa1fc8cU, 0xcc68839eU, 0x6d69fa49U, 0xb9cd5d1cU,
      0x67a6ac4eU, 0x066a60cbU, 0x3da7e149U, 0xa6307173U},
     0x1.3d06d19919f95p-1, 0x1.c5dcdb949fa68p-4,
     0.531216766543139, -0.44401303731530245},
    {{UINT64_C(1), 0, -2, 3},
     {0x785d785dU, 0x1cff6fe0U, 0x41e90a68U, 0x165ecf5dU,
      0xdd8081f8U, 0x03e5ee83U, 0xe181ccc0U, 0x3a597ccfU},
     0x1.c0dffe38baf0bp-1, 0x1.773d7859a02bap-2,
     -0.2423454094477596, -0.2699054194836502},
    {{UINT64_C(73452342811), 0, 0, -7},
     {0x09bfd9abU, 0xd1e9dfa4U, 0xad5eba6eU, 0xbf801822U,
      0x2e31348dU, 0x84ca5df4U, 0xf70fd757U, 0x4cda298cU},
     0x1.49bfd3a357b37p-1, 0x1.10c405fb75d2cp-3,
     0.4443018851434282, -0.49253194931734035},
    {{UINT64_C(9223372036854775813), 1, 2, -3},
     {0x9fac9e26U, 0x14c11896U, 0x8e60a3c8U, 0xf50d02e8U,
      0x7edc1c9cU, 0x006857c5U, 0x9d924082U, 0xda95bb21U},
     0x1.2c3182284d3d5p-1, 0x1.d0041beb9146dp-1,
     0.6076207005799297, -0.40582607274461213},
    {{UINT64_C(18446744073709551615), 2147483647, -2147483647, 1},
     {0xd0627e59U, 0x0751da1dU, 0x1eaf2dfdU, 0x194104bfU,
      0x89265f85U, 0xa45e16e8U, 0x6cffb023U, 0x6f75771eU},
     0x1.dda5107597e68p-4, 0x1.7e088233fa5b5p-1,
     -0.03537755492436404, -1.4654853367231795},
    {{UINT64_C(1), 2, -1, 0},
     {0xc5af956fU, 0xabc33944U, 0xc9fc8201U, 0xf8a8c399U,
      0x9bff7804U, 0x9d53f368U, 0x0d36c152U, 0x71de3ee8U},
     0x1.10e70eadbe56ap-2, 0x1.338751f00305fp-1,
     -0.9275860720458982, -0.6796624513894655},
    {{UINT64_C(17), 1, -2, 0},
     {0x26829fc0U, 0x15bfd642U, 0xd0b23fc0U, 0x9bf2f6a4U,
      0xb41583a3U, 0x3162e11cU, 0xce387e3cU, 0x45bc84bdU},
     0x1.0b5afc57027e2p-2, 0x1.49ede537807f7p-1,
     -0.7137429906377667, -0.9129424084945994},
}};

/**
 * @brief Fail a deterministic gate without changing its predeclared tolerance.
 * @param condition Whether the fixed contract holds.
 * @param message Scientific/protocol gate identifying a failure.
 */
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

/**
 * @brief Check a finite Gaussian component against the independent Python value.
 * @param actual Host or execution-space component.
 * @param expected Independent Python Box-Muller component.
 * @param label Gate label, including backend and vector number.
 */
void close(double actual, double expected, const std::string& label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > GaussianTolerance) {
        std::cerr << std::setprecision(17) << label << ": " << actual << " != " << expected << '\n';
        throw std::runtime_error(label);
    }
}

/**
 * @brief Compare all digest bits, both uniforms and the complex Gaussian to the oracle.
 * @param digest Produced canonical SHA256 digest.
 * @param u1 First open uniform.
 * @param u2 Second open uniform.
 * @param value Produced Gaussian at the requested signed mode.
 * @param oracle Independent Python fixed vector.
 * @param label Execution context for failures.
 */
void checkOracle(const cosmology::icRandomDetail::ModeDigest& digest,
                 double u1, double u2, const Kokkos::complex<double>& value,
                 const Oracle& oracle, const std::string& label) {
    for (unsigned word = 0; word < 8; ++word)
        require(digest.words_m[word] == oracle.words_m[word], label + ": exact SHA256 digest");
    const auto& input = oracle.input_m;
    const auto first = input.x_m != 0 ? input.x_m : (input.y_m != 0 ? input.y_m : input.z_m);
    require(digest.conjugate_m == (first < 0), label + ": canonical conjugation flag");
    require(u1 == oracle.u1_m && u2 == oracle.u2_m, label + ": exact 52-bit uniform encoding");
    require(u1 > 0 && u1 < 1 && u2 > 0 && u2 < 1, label + ": open uniform interval");
    close(value.real(), oracle.real_m, label + ": Gaussian real component");
    close(value.imag(), oracle.imaginary_m, label + ": Gaussian imaginary component");
}

/** @brief Exercise fixed vectors on the host and in the configured Kokkos execution space. */
void runChecks() {
    constexpr int CaseCount = static_cast<int>(Oracles.size());
    using Complex = Kokkos::complex<double>;
    using Digest = cosmology::icRandomDetail::ModeDigest;
    Kokkos::View<ModeInput*> inputs("IC RNG oracle inputs", CaseCount);
    Kokkos::View<Digest*> digests("IC RNG device digests", CaseCount);
    Kokkos::View<double*[2]> uniforms("IC RNG device uniforms", CaseCount);
    Kokkos::View<Complex*> values("IC RNG device Gaussian", CaseCount);
    Kokkos::View<Complex*> opposite("IC RNG device conjugates", CaseCount);
    Kokkos::View<Complex*> repeated("IC RNG reordered repetition", CaseCount);
    auto hostInputs = Kokkos::create_mirror_view(inputs);
    for (int index = 0; index < CaseCount; ++index) {
        const auto& oracle = Oracles[index];
        const auto& input = oracle.input_m;
        hostInputs(index) = input;
        const auto digest = cosmology::icRandomDetail::modeDigest(
            input.seed_m, input.x_m, input.y_m, input.z_m);
        const auto value = cosmology::modeGaussian(input.seed_m, input.x_m, input.y_m, input.z_m);
        checkOracle(digest, cosmology::icRandomDetail::modeUniform(digest, 0U),
                    cosmology::icRandomDetail::modeUniform(digest, 1U), value, oracle,
                    "host vector " + std::to_string(index));
    }
    Kokkos::deep_copy(inputs, hostInputs);
    Kokkos::parallel_for("IC RNG fixed-vector regression", CaseCount, KOKKOS_LAMBDA(int index) {
        const auto input = inputs(index);
        const auto digest = cosmology::icRandomDetail::modeDigest(
            input.seed_m, input.x_m, input.y_m, input.z_m);
        digests(index) = digest;
        uniforms(index, 0) = cosmology::icRandomDetail::modeUniform(digest, 0U);
        uniforms(index, 1) = cosmology::icRandomDetail::modeUniform(digest, 1U);
        values(index) = cosmology::modeGaussian(input.seed_m, input.x_m, input.y_m, input.z_m);
        opposite(index) = cosmology::modeGaussian(input.seed_m, -input.x_m, -input.y_m, -input.z_m);
    });
    Kokkos::parallel_for("IC RNG reversed evaluation order", CaseCount, KOKKOS_LAMBDA(int counter) {
        const int index = CaseCount - 1 - counter;
        const auto input = inputs(index);
        repeated(index) = cosmology::modeGaussian(input.seed_m, input.x_m, input.y_m, input.z_m);
    });
    const auto hostDigests = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), digests);
    const auto hostUniforms = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uniforms);
    const auto hostValues = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    const auto hostOpposite = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), opposite);
    const auto hostRepeated = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), repeated);
    for (int index = 0; index < CaseCount; ++index) {
        const auto label = "device vector " + std::to_string(index);
        checkOracle(hostDigests(index), hostUniforms(index, 0), hostUniforms(index, 1),
                    hostValues(index), Oracles[index], label);
        require(hostOpposite(index).real() == hostValues(index).real()
                && hostOpposite(index).imag() == -hostValues(index).imag(),
                label + ": exact Hermitian conjugation");
        require(hostRepeated(index).real() == hostValues(index).real()
                && hostRepeated(index).imag() == hostValues(index).imag(),
                label + ": stateless reordered repetition");
    }
    require(hostValues(1) != hostValues(7), "Changing the seed must change the fixed test draw");
    require(hostValues(1) != hostValues(8), "Cartesian mode order must enter the fixed key");
    std::cout << "PASS: " << CaseCount
              << " independent Python vectors; exact SHA256/uniforms, host/device Gaussian,"
                 " Hermitian symmetry, high unsigned seeds and reordered repetition; backend="
              << Kokkos::DefaultExecutionSpace::name() << '\n';
}

} // namespace

/**
 * @brief Run this local RNG test using the CMake-selected Kokkos backend.
 * @param argc Number of process arguments forwarded to Kokkos.
 * @param argv Process arguments, including any Kokkos backend options.
 * @return Zero only when all fixed host and execution-space checks pass.
 */
int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    try {
        runChecks();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
