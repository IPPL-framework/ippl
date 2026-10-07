/** @file PhaseSpaceIO.h
 * @brief Portable, versioned phase-space wire format used before parallel HDF5 qualification.
 * @ingroup cosmology_core
 * @see cosmology_contracts cosmology_quijote
 * Host-only encoding: x in comoving Mpc/h, p=a*v_pec/100 in Mpc/h.
 * A 128-byte little-endian header precedes 56-byte (uint64 ID, six float64) records.
 */
#ifndef IPPL_COSMOLOGY_PHASE_SPACE_IO_H
#define IPPL_COSMOLOGY_PHASE_SPACE_IO_H

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace cosmology {
constexpr std::size_t PhaseHeaderBytes = 128; ///< Fixed v1 header size, including zero reserved bytes.
constexpr std::size_t PhaseRecordBytes = 56; ///< One uint64 ID and six IEEE float64 coordinates.
constexpr std::size_t PhaseChunkRecords = 65536; ///< Bounded serial I/O/MPI staging batch.

/** @brief Epoch, background and partition metadata for equal-mass phase space. */
struct PhaseSpaceHeader {
    std::uint64_t count = 0; ///< File-local particle count, including empty snapshot shards.
    std::uint64_t total = 0; ///< Exact positive global particle count.
    double a = 0; ///< Synchronized epoch, a=1/(1+z).
    double box = 0; ///< Comoving periodic side in Mpc/h.
    double omegaMatter = 0; ///< Present-day total matter fraction.
    double omegaLambda = 0; ///< Present-day cosmological constant fraction.
    double hubble = 0; ///< Dimensionless H0/(100 km/s/Mpc).
    double particleMass = 0; ///< Physical equal particle mass in Msun/h; solver uses unit weights.
    std::uint64_t flags = 0; ///< Bit zero: one complete file ordered by IDs 0,...,N-1.
};

/** @brief Read an unsigned integer from eight little-endian bytes, independent of host byte order.
 * @param bytes Address of eight serialized bytes.
 * @return Decoded unsigned value.
 */
inline std::uint64_t phaseReadInteger(const char* bytes) {
    std::uint64_t result = 0;
    for (unsigned i = 0; i < 8; ++i)
        result |= std::uint64_t(static_cast<unsigned char>(bytes[i])) << (8 * i);
    return result;
}
/** @brief Encode an unsigned integer as eight little-endian bytes.
 * @param bytes Writable address of eight destination bytes.
 * @param value Unsigned value to serialize without narrowing.
 */
inline void phaseWriteInteger(char* bytes, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>((value >> (8 * i)) & 255);
}
/** @brief Decode IEEE float64 while preserving its serialized value exactly.
 * @param bytes Address of eight serialized bytes.
 * @return Decoded double; physical finiteness is checked by the caller.
 */
inline double phaseReadDouble(const char* bytes) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    const auto bits = phaseReadInteger(bytes);
    double result;
    std::memcpy(&result, &bits, 8);
    return result;
}
/** @brief Encode IEEE float64 without narrowing or cosmological rescaling.
 * @param bytes Writable address of eight destination bytes.
 * @param value IEEE double to serialize exactly.
 */
inline void phaseWriteDouble(char* bytes, double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, 8);
    phaseWriteInteger(bytes, bits);
}
/** @brief Reject unsupported partition flags, nonphysical headers and unsafe byte counts.
 * @param header Equal-mass flat-background partition metadata.
 */
inline void validatePhaseHeader(const PhaseSpaceHeader& header) {
    if (header.total == 0 || header.count > header.total || header.flags > 1
        || (header.flags == 1 && header.count != header.total)
        || header.count > (std::uint64_t(std::numeric_limits<std::int64_t>::max()) - PhaseHeaderBytes) / PhaseRecordBytes
        || !std::isfinite(header.a) || header.a <= 0 || header.a > 1
        || !std::isfinite(header.box) || header.box <= 0
        || !std::isfinite(header.omegaMatter) || header.omegaMatter <= 0 || header.omegaMatter > 1
        || !std::isfinite(header.omegaLambda) || header.omegaLambda < 0
        || std::abs(header.omegaMatter + header.omegaLambda - 1) > 1.e-10
        || !std::isfinite(header.hubble) || header.hubble <= 0
        || !std::isfinite(header.particleMass) || header.particleMass <= 0)
        throw std::invalid_argument("Invalid canonical phase-space header");
}
/** @brief Read and validate the v1 header, including magic and zero reserved bytes.
 * @param input Binary stream positioned at its initial header byte.
 * @return Validated epoch, background, counts and equal particle mass.
 */
inline PhaseSpaceHeader readPhaseHeader(std::istream& input) {
    std::array<char, PhaseHeaderBytes> bytes{};
    if (!input.read(bytes.data(), bytes.size())) throw std::runtime_error("Truncated phase-space header");
    if (std::memcmp(bytes.data(), "IPPLPS01", 8) != 0)
        throw std::invalid_argument("Unknown phase-space magic/version");
    for (std::size_t i = 80; i < bytes.size(); ++i)
        if (bytes[i] != 0) throw std::invalid_argument("Nonzero reserved phase-space header bytes");
    PhaseSpaceHeader header;
    header.count = phaseReadInteger(bytes.data() + 8);
    header.total = phaseReadInteger(bytes.data() + 16);
    header.a = phaseReadDouble(bytes.data() + 24);
    header.box = phaseReadDouble(bytes.data() + 32);
    header.omegaMatter = phaseReadDouble(bytes.data() + 40);
    header.omegaLambda = phaseReadDouble(bytes.data() + 48);
    header.hubble = phaseReadDouble(bytes.data() + 56);
    header.particleMass = phaseReadDouble(bytes.data() + 64);
    header.flags = phaseReadInteger(bytes.data() + 72);
    validatePhaseHeader(header);
    return header;
}
/** @brief Write the v1 header with zero reserved bytes; no native struct padding is serialized.
 * @param output Binary destination stream.
 * @param header Valid equal-mass partition metadata.
 */
inline void writePhaseHeader(std::ostream& output, const PhaseSpaceHeader& header) {
    validatePhaseHeader(header);
    std::array<char, PhaseHeaderBytes> bytes{};
    std::memcpy(bytes.data(), "IPPLPS01", 8);
    phaseWriteInteger(bytes.data() + 8, header.count);
    phaseWriteInteger(bytes.data() + 16, header.total);
    phaseWriteDouble(bytes.data() + 24, header.a);
    phaseWriteDouble(bytes.data() + 32, header.box);
    phaseWriteDouble(bytes.data() + 40, header.omegaMatter);
    phaseWriteDouble(bytes.data() + 48, header.omegaLambda);
    phaseWriteDouble(bytes.data() + 56, header.hubble);
    phaseWriteDouble(bytes.data() + 64, header.particleMass);
    phaseWriteInteger(bytes.data() + 72, header.flags);
    output.write(bytes.data(), bytes.size());
    if (!output) throw std::runtime_error("Phase-space header write failed");
}
} // namespace cosmology
#endif
