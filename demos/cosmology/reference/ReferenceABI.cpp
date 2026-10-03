// Build manifest helper; the supplied reference source is never edited.
#include "TypesAndDefs.h"
#include <cstdint>
#include <iostream>

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
