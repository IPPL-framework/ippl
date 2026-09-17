/**
 * @file HeaderIncludeOrder.cpp
 * @brief Verifies that Manager/datatypes.h compiles without a preceding Ippl.h include.
 * 
 * This small test verifies that we don't get a compiler error in OPALX when including 
 * `datatypes.h` without first including `Ippl.h`. 
 */

#include "Manager/datatypes.h"

int main() {
    return 0;
}
