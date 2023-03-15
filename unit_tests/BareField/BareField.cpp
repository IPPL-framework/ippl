//
// Unit test BareFieldTest
//   Test the functionality of the class BareField.
//
// Copyright (c) 2021 Paul Scherrer Institut, Villigen PSI, Switzerland
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
#include "Ippl.h"

#include <cmath>
#include "gtest/gtest.h"

class BareFieldTest : public ::testing::Test {
public:
    static constexpr size_t dim = 3;
    typedef ippl::BareField<double, dim> field_type;
    typedef ippl::BareField<ippl::Vector<double, 3>, dim> vfield_type;

    BareFieldTest()
        : nPoints(8) {
        setup();
    }

    void setup() {
        ippl::Index I(nPoints);
        ippl::NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag domDec[dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++)
            domDec[d] = ippl::PARALLEL;

        ippl::FieldLayout<dim> layout(owned, domDec);
        field  = std::make_unique<field_type>(layout);
        vfield = std::make_unique<vfield_type>(layout);
    }

    std::unique_ptr<field_type> field;
    std::unique_ptr<vfield_type> vfield;
    size_t nPoints;
};

TEST_F(BareFieldTest, Sum) {
    double val = 1.0;

    *field = val;

    double sum = field->sum();

    ASSERT_DOUBLE_EQ(val * std::pow(nPoints, dim), sum);
}

TEST_F(BareFieldTest, Min) {
    const ippl::NDIndex<dim> lDom = field->getLayout().getLocalNDIndex();
    const int shift               = field->getNghost();

    auto view   = field->getView();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                const size_t ig = i + lDom[0].first();
                const size_t jg = j + lDom[1].first();
                const size_t kg = k + lDom[2].first();

                mirror(i, j, k) = -1.0 + (ig + jg + kg);
            }
        }
    }
    Kokkos::deep_copy(view, mirror);

    double min = field->min();
    // minimum value -1 + nghost + nghost + nghost
    ASSERT_DOUBLE_EQ(min, 2.);
}

TEST_F(BareFieldTest, Max) {
    const ippl::NDIndex<dim> lDom = field->getLayout().getLocalNDIndex();
    const int shift               = field->getNghost();

    auto view   = field->getView();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                const size_t ig = i + lDom[0].first();
                const size_t jg = j + lDom[1].first();
                const size_t kg = k + lDom[2].first();

                mirror(i, j, k) = -1.0 + (ig + jg + kg);
            }
        }
    }
    Kokkos::deep_copy(view, mirror);

    double max      = field->max();
    double expected = -1. + nPoints * 3;
    ASSERT_DOUBLE_EQ(max, expected);
}

TEST_F(BareFieldTest, Prod) {
    *field     = 2.;
    double val = field->prod();
    ASSERT_DOUBLE_EQ(val, pow(2, nPoints * nPoints * nPoints));
}

TEST_F(BareFieldTest, ScalarMultiplication) {
    *field = 1.;
    *field = *field * 10;

    const int shift = field->getNghost();

    auto view   = field->getView();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                ASSERT_DOUBLE_EQ(mirror(i, j, k), 10.);
            }
        }
    }
}

TEST_F(BareFieldTest, DotProduct) {
    *vfield = 1.;
    *field  = 5. * dot(*vfield, *vfield);

    const int shift = field->getNghost();

    auto view   = field->getView();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                ASSERT_DOUBLE_EQ(mirror(i, j, k), 15.);
            }
        }
    }
}

TEST_F(BareFieldTest, AllFuncs) {
    double pi    = acos(-1.);
    double alpha = pi / 4;
    *field       = alpha;
    // Compute new value
    double beta =
        fabs(7.0 * (sin(alpha) * cos(alpha)) / (tan(alpha) * acos(alpha)) - exp(alpha) + erf(alpha)
             + (asin(alpha) * cosh(alpha)) / (atan(alpha) * sinh(alpha)) + tanh(alpha) * log(alpha)
             - log10(alpha) * sqrt(alpha) + floor(alpha) * ceil(alpha));

    // Compute same value via field ops
    *field = fabs(7.0 * (sin(*field) * cos(*field)) / (tan(*field) * acos(*field)) - exp(*field)
                  + erf(*field) + (asin(*field) * cosh(*field)) / (atan(*field) * sinh(*field))
                  + tanh(*field) * log(*field) - log10(*field) * sqrt(*field)
                  + floor(*field) * ceil(*field));

    const int shift = field->getNghost();

    auto view   = field->getView();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                ASSERT_DOUBLE_EQ(mirror(i, j, k), beta);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
