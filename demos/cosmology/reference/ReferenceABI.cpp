/** @file ReferenceABI.cpp
 * @brief Host probe of the supplied original initializer ABI/precision layout.
 * @ingroup cosmology_reference
 * @see cosmology_contracts cosmology_validation cosmology_references
 */
// Build manifest helper; the supplied reference source is never edited.
#include "TypesAndDefs.h"
#include <cstdint>
#include <iostream>

/**
 * @brief Run host probe of the supplied original initializer abi/precision layout.
 * @see cosmology_contracts cosmology_validation
 * @return Zero on successful completion; malformed/native fatal errors return nonzero or abort the communicator.
 * Fatal distributed failures must terminate communicator peers; the host-only test uses ordinary process status.
 */
int main() {
    const std::uint32_t marker = 1;
    const bool little = *reinterpret_cast<const unsigned char*>(&marker) == 1;
    std::cout << "sizeof_real=" << sizeof(initializer::real)
              << "\nsizeof_integer=" << sizeof(initializer::integer)
              << "\nsizeof_IDtype=" << sizeof(initializer::IDtype)
              << "\nendian=" << (little ? "little" : "big")
              << "\nbinary_record_bytes=" << 6 * sizeof(initializer::real)
                                                   + sizeof(initializer::integer) << '\n';
    return 0;
}
