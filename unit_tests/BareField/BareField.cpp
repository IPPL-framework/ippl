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

#include "MultirankUtils.h"
#include "gtest/gtest.h"

class BareFieldTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using field_type = ippl::BareField<double, Dim>;

    template <unsigned Dim>
    using vfield_type = ippl::BareField<ippl::Vector<double, Dim>, Dim>;

    BareFieldTest() {
        computeGridSizes(nPoints);
        setup(this);
    }

    template <size_t Idx, unsigned Dim>
    void setupDim() {
        std::array<ippl::Index, Dim> indices;
        for (unsigned d = 0; d < Dim; d++) {
            indices[d] = ippl::Index(nPoints[d]);
        }
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(indices);

        ippl::e_dim_tag domDec[Dim];
        for (auto& tag : domDec) {
            tag = ippl::PARALLEL;
        }

        std::get<Idx>(layouts) = ippl::FieldLayout<Dim>(owned, domDec);
        auto& layout           = std::get<Idx>(layouts);

        std::get<Idx>(fields)  = std::make_shared<field_type<Dim>>(layout);
        std::get<Idx>(vfields) = std::make_shared<vfield_type<Dim>>(layout);
    }

    Collection<ippl::FieldLayout> layouts;

    PtrCollection<std::shared_ptr, field_type> fields;
    PtrCollection<std::shared_ptr, vfield_type> vfields;
    size_t nPoints[MaxDim];
};

template <unsigned Dim>
struct FieldVal {
    const typename BareFieldTest::field_type<Dim>::view_type view;
    const ippl::NDIndex<Dim> lDom;

    template <typename... Idx>
    KOKKOS_INLINE_FUNCTION void operator()(const Idx... args) const {
        double tot = (args + ...);
        for (unsigned d = 0; d < Dim; d++) {
            tot += lDom[d].first();
        }
        view(args...) = tot - 1;
    }
};

TEST_F(BareFieldTest, Sum) {
    double val              = 1.0;
    double expected[MaxDim] = {val * nPoints[0]};
    for (unsigned d = 1; d < MaxDim; d++) {
        expected[d] = expected[d - 1] * nPoints[d];
    }

    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field     = val;
        double sum = field->sum();
        ASSERT_DOUBLE_EQ(expected[dimToIndex(Dim)], sum);
    };

    apply(check, fields);
}

TEST_F(BareFieldTest, Min) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();
        auto view                     = field->getView();

        Kokkos::parallel_for("Set field", field->getFieldRangePolicy(), FieldVal<Dim>{view, lDom});
        Kokkos::fence();

        double min = field->min();
        // minimum value in 3D: -1 + nghost + nghost + nghost
        ASSERT_DOUBLE_EQ(min, field->getNghost() * Dim - 1);
    };

    apply(check, fields);
}

TEST_F(BareFieldTest, Max) {
    double expected[MaxDim] = {nPoints[0] - 1.};
    for (unsigned d = 1; d < MaxDim; d++) {
        expected[d] = expected[d - 1] + nPoints[d];
    }
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();
        auto view                     = field->getView();

        Kokkos::parallel_for("Set field", field->getFieldRangePolicy(), FieldVal<Dim>{view, lDom});
        Kokkos::fence();

        double max = field->max();
        ASSERT_DOUBLE_EQ(max, expected[dimToIndex(Dim)]);
    };

    apply(check, fields);
}

TEST_F(BareFieldTest, Prod) {
    double sizes[MaxDim] = {(double)nPoints[0]};
    for (unsigned d = 1; d < MaxDim; d++) {
        sizes[d] = sizes[d - 1] * nPoints[d];
    }
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field     = 2.;
        double val = field->prod();
        ASSERT_DOUBLE_EQ(val, pow(2, sizes[dimToIndex(Dim)]));
    };

    apply(check, fields);
}

TEST_F(BareFieldTest, ScalarMultiplication) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field = 1.;
        *field = *field * 10;

        const int shift = field->getNghost();

        auto view   = field->getView();
        auto mirror = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(mirror, view);

        nestedViewLoop<Dim>(mirror, shift, [&]<typename... Idx>(const Idx... args) {
            ASSERT_DOUBLE_EQ(mirror(args...), 10.);
        });
    };

    apply(check, fields);
}

TEST_F(BareFieldTest, DotProduct) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field,
                                   std::shared_ptr<vfield_type<Dim>>& vfield) {
        *vfield = 1.;
        *field  = 5. * dot(*vfield, *vfield);

        const int shift = field->getNghost();

        auto view   = field->getView();
        auto mirror = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(mirror, view);

        nestedViewLoop<Dim>(mirror, shift, [&]<typename... Idx>(const Idx... args) {
            ASSERT_DOUBLE_EQ(mirror(args...), 5 * Dim);
        });
    };

    apply(check, fields, vfields);
}

TEST_F(BareFieldTest, AllFuncs) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        double pi    = acos(-1.);
        double alpha = pi / 4;
        *field       = alpha;
        // Compute new value
        double beta = fabs(7.0 * (sin(alpha) * cos(alpha)) / (tan(alpha) * acos(alpha)) - exp(alpha)
                           + erf(alpha) + (asin(alpha) * cosh(alpha)) / (atan(alpha) * sinh(alpha))
                           + tanh(alpha) * log(alpha) - log10(alpha) * sqrt(alpha)
                           + floor(alpha) * ceil(alpha));

        // Compute same value via field ops
        *field = fabs(7.0 * (sin(*field) * cos(*field)) / (tan(*field) * acos(*field)) - exp(*field)
                      + erf(*field) + (asin(*field) * cosh(*field)) / (atan(*field) * sinh(*field))
                      + tanh(*field) * log(*field) - log10(*field) * sqrt(*field)
                      + floor(*field) * ceil(*field));

        const int shift = field->getNghost();

        auto view   = field->getView();
        auto mirror = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(mirror, view);

        nestedViewLoop<Dim>(mirror, shift, [&]<typename... Idx>(const Idx... args) {
            ASSERT_DOUBLE_EQ(mirror(args...), beta);
        });
    };

    apply(check, fields);
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
