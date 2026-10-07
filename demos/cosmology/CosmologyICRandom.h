/**
 * @file CosmologyICRandom.h
 * @brief Stateless, resolution-independent Gaussian Fourier draws for optional native ICs.
 * @ingroup cosmology_core
 * @see cosmology_numerics cosmology_contracts cosmology_validation
 *
 * The byte protocol matches python/gaussian_fixture.py, RngDomain and
 * mode_gaussian. It is independent of mesh dimensions, MPI ownership, retained
 * modes and execution order. It does not replace or modify the legacy RNG.
 */
#ifndef IPPL_COSMOLOGY_IC_RANDOM_H
#define IPPL_COSMOLOGY_IC_RANDOM_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdint>

namespace cosmology {
namespace icRandomDetail {

/**
 * @brief SHA256 state words of the canonical mode key and its conjugation flag.
 *
 * Words are the standard SHA256 digest's eight big-endian uint32 words.
 * This internal value is exposed to the small fixed-vector regression so the
 * integer protocol is checked exactly, separately from device transcendental
 * roundoff. No allocation or shared state is involved.
 */
struct ModeDigest {
    std::uint32_t words_m[8]; ///< SHA256 digest, serialized big-endian word by word.
    bool conjugate_m; ///< True if the requested mode is the negative canonical member.
};

/**
 * @brief Rotate a 32-bit SHA256 word right, with arithmetic modulo 2^32.
 * @param value Unsigned word to rotate.
 * @param count Fixed SHA256 rotation count, strictly between zero and 32.
 * @return Rotated unsigned word.
 */
KOKKOS_INLINE_FUNCTION std::uint32_t rotateRight(std::uint32_t value, unsigned count) {
    return (value >> count) | (value << (32U - count));
}

/**
 * @brief Hash the exact one-block signed-mode message with SHA256.
 *
 * The canonical member has its first nonzero Cartesian component positive.
 * The message is the ASCII domain "IPPL-band-limited-Gaussian-v1" including
 * its terminating NUL, then a little-endian uint64 seed and three little-endian
 * int32 canonical components. SHA256 padding and compression use their standard
 * big-endian encoding; every arithmetic overflow is unsigned modulo 2^32.
 *
 * Execution and memory are local to the calling CPU/GPU thread. The fixed
 * stack arrays require no host/device copies, atomics, pool or rank seed.
 *
 * @param seed Full unsigned 64-bit realization seed.
 * @param kx Signed physical integer x mode, not a wrapped FFT storage index.
 * @param ky Signed physical integer y mode.
 * @param kz Signed physical integer z mode.
 * @return Canonical digest and whether the requested result must be conjugated.
 * @pre The mode is nonzero and no component equals INT32_MIN. Invalid calls
 * abort on host or device; callers exclude DC before requesting a random draw.
 */
KOKKOS_INLINE_FUNCTION ModeDigest modeDigest(std::uint64_t seed, std::int32_t kx,
                                             std::int32_t ky, std::int32_t kz) {
    constexpr std::int32_t MinimumComponent = -2147483647 - 1;
    if ((kx == 0 && ky == 0 && kz == 0) || kx == MinimumComponent
        || ky == MinimumComponent || kz == MinimumComponent) {
        Kokkos::abort("Cosmology mode RNG requires a nonzero, conjugatable int32 mode");
    }
    const std::int32_t first = kx != 0 ? kx : (ky != 0 ? ky : kz);
    const bool conjugate = first < 0;
    const std::int32_t components[3] = {
        conjugate ? -kx : kx, conjugate ? -ky : ky, conjugate ? -kz : kz};
    constexpr char Domain[] = "IPPL-band-limited-Gaussian-v1";
    constexpr unsigned DomainBytes = sizeof(Domain);
    constexpr unsigned MessageBytes = DomainBytes + 8U + 3U * 4U;
    static_assert(MessageBytes <= 55U, "Mode key must fit a single SHA256 block");
    unsigned char block[64] = {};
    for (unsigned index = 0; index < DomainBytes; ++index)
        block[index] = static_cast<unsigned char>(Domain[index]);
    for (unsigned byte = 0; byte < 8; ++byte)
        block[DomainBytes + byte] = static_cast<unsigned char>(seed >> (8U * byte));
    for (unsigned component = 0; component < 3; ++component) {
        // Signed-to-unsigned conversion specifies two's-complement bytes modulo 2^32.
        const auto encoded = static_cast<std::uint32_t>(components[component]);
        for (unsigned byte = 0; byte < 4; ++byte)
            block[DomainBytes + 8U + 4U * component + byte] =
                static_cast<unsigned char>(encoded >> (8U * byte));
    }
    block[MessageBytes] = 0x80U;
    constexpr std::uint64_t MessageBits = MessageBytes * 8ULL;
    for (unsigned byte = 0; byte < 8; ++byte)
        block[63U - byte] = static_cast<unsigned char>(MessageBits >> (8U * byte));

    std::uint32_t schedule[64];
    for (unsigned index = 0; index < 16; ++index) {
        schedule[index] = (std::uint32_t(block[4U * index]) << 24U)
                        | (std::uint32_t(block[4U * index + 1U]) << 16U)
                        | (std::uint32_t(block[4U * index + 2U]) << 8U)
                        | std::uint32_t(block[4U * index + 3U]);
    }
    for (unsigned index = 16; index < 64; ++index) {
        const auto left = schedule[index - 15U];
        const auto right = schedule[index - 2U];
        const auto sigma0 = rotateRight(left, 7U) ^ rotateRight(left, 18U) ^ (left >> 3U);
        const auto sigma1 = rotateRight(right, 17U) ^ rotateRight(right, 19U) ^ (right >> 10U);
        schedule[index] = schedule[index - 16U] + sigma0 + schedule[index - 7U] + sigma1;
    }
    constexpr std::uint32_t RoundConstants[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    constexpr std::uint32_t InitialState[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    auto a = InitialState[0], b = InitialState[1], c = InitialState[2], d = InitialState[3];
    auto e = InitialState[4], f = InitialState[5], g = InitialState[6], h = InitialState[7];
    for (unsigned index = 0; index < 64; ++index) {
        const auto sigma1 = rotateRight(e, 6U) ^ rotateRight(e, 11U) ^ rotateRight(e, 25U);
        const auto choice = (e & f) ^ ((~e) & g);
        const auto firstTerm = h + sigma1 + choice + RoundConstants[index] + schedule[index];
        const auto sigma0 = rotateRight(a, 2U) ^ rotateRight(a, 13U) ^ rotateRight(a, 22U);
        const auto majority = (a & b) ^ (a & c) ^ (b & c);
        const auto secondTerm = sigma0 + majority;
        h = g; g = f; f = e; e = d + firstTerm;
        d = c; c = b; b = a; a = firstTerm + secondTerm;
    }
    return {{InitialState[0] + a, InitialState[1] + b, InitialState[2] + c,
             InitialState[3] + d, InitialState[4] + e, InitialState[5] + f,
             InitialState[6] + g, InitialState[7] + h}, conjugate};
}

/**
 * @brief Reverse the bytes of a 32-bit SHA256 digest word without host-endian assumptions.
 * @param word One standard big-endian digest word.
 * @return The integer obtained by interpreting its four serialized bytes little-endian.
 */
KOKKOS_INLINE_FUNCTION std::uint32_t reverseBytes(std::uint32_t word) {
    return (word >> 24U) | ((word >> 8U) & 0x0000ff00U)
         | ((word << 8U) & 0x00ff0000U) | (word << 24U);
}

/**
 * @brief Recover one open-interval 52-bit uniform from eight consecutive digest bytes.
 * @param digest SHA256 state words from modeDigest.
 * @param index Either zero or one, selecting bytes [0,8) or [8,16).
 * @return Exactly representable (uint52 + 1/2)/2^52, strictly between zero and one.
 */
KOKKOS_INLINE_FUNCTION double modeUniform(const ModeDigest& digest, unsigned index) {
    const auto low = std::uint64_t(reverseBytes(digest.words_m[2U * index]));
    const auto high = std::uint64_t(reverseBytes(digest.words_m[2U * index + 1U]));
    const auto bits = low | (high << 32U);
    constexpr double UniformDenominator = 4503599627370496.0; // Exact 2^52 protocol constant.
    return (static_cast<double>(bits >> 12U) + 0.5) / UniformDenominator;
}

} // namespace icRandomDetail

/**
 * @brief Draw one unit-variance complex Gaussian keyed by seed and physical integer mode.
 *
 * For independent open uniforms u1,u2 from the fixed SHA256 message, return
 * @f$g=\sqrt{-\ln u_1}\,[\cos(2\pi u_2)+i\sin(2\pi u_2)]@f$ for the
 * canonical member and its exact conjugate for the opposite mode. Thus
 * @f$E[g]=0@f$, @f$E[|g|^2]=1@f$, and real/imaginary variances are 1/2.
 * The caller forms @f$\delta_{\boldsymbol m}=\sqrt{P(k)/L^3}\,g@f$ and
 * applies the lattice's half-cell sampling phase separately. No realization
 * normalization, growth factor, cutoff or grid dimension enters this draw.
 *
 * Integer digest/uniform results match the frozen Python protocol exactly.
 * Host/device log, sqrt and sin/cos can differ at roundoff; this interface does
 * not promise bitwise particle agreement across FFT/back-end implementations.
 * Execution/memory are per-thread, with no shared RNG state or data movement.
 *
 * @param seed Full unsigned 64-bit realization seed.
 * @param kx Signed physical integer x mode.
 * @param ky Signed physical integer y mode.
 * @param kz Signed physical integer z mode.
 * @return Unit-variance Kokkos complex Gaussian coefficient before physical scaling.
 * @pre Nonzero mode with all components in [-2147483647,2147483647].
 * @see icRandomDetail::modeDigest
 */
KOKKOS_INLINE_FUNCTION Kokkos::complex<double> modeGaussian(
    std::uint64_t seed, std::int32_t kx, std::int32_t ky, std::int32_t kz) {
    const auto digest = icRandomDetail::modeDigest(seed, kx, ky, kz);
    const double u1 = icRandomDetail::modeUniform(digest, 0U);
    const double u2 = icRandomDetail::modeUniform(digest, 1U);
    const double radius = Kokkos::sqrt(-Kokkos::log(u1));
    const double angle = 2 * Kokkos::numbers::pi_v<double> * u2;
    const double real = radius * Kokkos::cos(angle);
    const double imaginary = radius * Kokkos::sin(angle);
    return {real, digest.conjugate_m ? -imaginary : imaginary};
}

} // namespace cosmology
#endif
