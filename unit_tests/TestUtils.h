//
// Utilities for versatile unit testing
//
// Copyright (c) 2023 Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <type_traits>

#include "MultirankUtils.h"
#include "gtest/gtest.h"

template <typename T>
void assertTypeParam(T valA, T valB) {
    if constexpr (std::is_same_v<T, double>) {
        ASSERT_DOUBLE_EQ(valA, valB);
    } else {
        ASSERT_FLOAT_EQ(valA, valB);
    }
};

#endif
