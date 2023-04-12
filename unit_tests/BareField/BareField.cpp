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

    BareFieldTest()
        : nPoints(8) {
        setup(this);
    }

    template <size_t Idx, unsigned Dim>
    void setupDim() {
        ippl::Index I(nPoints);
        std::array<ippl::Index, Dim> indices;
        indices.fill(I);
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(indices);

        ippl::e_dim_tag domDec[Dim];
        for (auto& tag : domDec)
            tag = ippl::PARALLEL;

        std::get<Idx>(layouts) = ippl::FieldLayout<Dim>(owned, domDec);
        auto& layout           = std::get<Idx>(layouts);

        std::get<Idx>(fields)  = std::make_shared<field_type<Dim>>(layout);
        std::get<Idx>(vfields) = std::make_shared<vfield_type<Dim>>(layout);
    }

    Collection<ippl::FieldLayout> layouts;

    PtrCollection<std::shared_ptr, field_type> fields;
    PtrCollection<std::shared_ptr, vfield_type> vfields;
    size_t nPoints;
};

TEST_F(BareFieldTest, Sum) {
    double val = 1.0;

    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field     = val;
        double sum = field->sum();
        ASSERT_DOUBLE_EQ(val * std::pow(nPoints, Dim), sum);
    };

    apply(check, fields);
}

TEST_F(BareFieldTest, Min) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();
        auto view                     = field->getView();

        using index_array_type = typename ippl::detail::RangePolicy<Dim>::index_array_type;
        Kokkos::parallel_for("Set field", field->getRangePolicy(),
                             ippl::detail::functorize<ippl::detail::FOR, Dim>(
                                 KOKKOS_LAMBDA(const index_array_type& args) {
                                     double tot = 0;
                                     for (unsigned d = 0; d < Dim; d++)
                                         tot += args[d] + lDom[d].first();
                                     ippl::apply<Dim>(view, args) = tot - 1;
                                 }));
        Kokkos::fence();

        double min = field->min();
        // minimum value -1 + nghost + nghost + nghost
        ASSERT_DOUBLE_EQ(min, field->getNghost() * Dim - 1);
    };

    apply(check, fields);
}

TEST_F(BareFieldTest, Max) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();
        auto view                     = field->getView();

        using index_array_type = typename ippl::detail::RangePolicy<Dim>::index_array_type;
        Kokkos::parallel_for("Set field", field->getRangePolicy(),
                             ippl::detail::functorize<ippl::detail::FOR, Dim>(
                                 KOKKOS_LAMBDA(const index_array_type& args) {
                                     double tot = 0;
                                     for (unsigned d = 0; d < Dim; d++)
                                         tot += args[d] + lDom[d].first();
                                     ippl::apply<Dim>(view, args) = tot - 1;
                                 }));
        Kokkos::fence();

        double max      = field->max();
        double expected = -1. + nPoints * Dim;
        ASSERT_DOUBLE_EQ(max, expected);
    };

    apply(check, fields);
}

TEST_F(BareFieldTest, Prod) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field     = 2.;
        double val = field->prod();
        ASSERT_DOUBLE_EQ(val, pow(2, pow(nPoints, Dim)));
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

    auto pair = zip(fields, vfields);
    apply(check, pair);
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
