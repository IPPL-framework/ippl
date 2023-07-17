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

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename T>
class BareFieldTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using field_type = ippl::BareField<T, Dim>;

    template <unsigned Dim>
    using vfield_type = ippl::BareField<ippl::Vector<T, Dim>, Dim>;

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

        std::get<Idx>(layouts) = ippl::FieldLayout<Dim>(MPI_COMM_WORLD, owned, domDec);
        auto& layout           = std::get<Idx>(layouts);

        std::get<Idx>(fields)  = std::make_shared<field_type<Dim>>(layout);
        std::get<Idx>(vfields) = std::make_shared<vfield_type<Dim>>(layout);
    }

    Collection<ippl::FieldLayout> layouts;

    PtrCollection<std::shared_ptr, field_type> fields;
    PtrCollection<std::shared_ptr, vfield_type> vfields;
    size_t nPoints[MaxDim];
};

template <typename T, unsigned Dim>
struct FieldVal {
    const typename BareFieldTest<T>::template field_type<Dim>::view_type view;
    const ippl::NDIndex<Dim> lDom;

    template <typename... Idx>
    KOKKOS_INLINE_FUNCTION void operator()(const Idx... args) const {
        T tot = (args + ...);
        for (unsigned d = 0; d < Dim; d++) {
            tot += lDom[d].first();
        }
        view(args...) = tot - 1;
    }
};

using Precisions = ::testing::Types<double, float>;

TYPED_TEST_CASE(BareFieldTest, Precisions);

TYPED_TEST(BareFieldTest, DeepCopy) {
    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            using view_type   = typename TestFixture::template field_type<Dim>::view_type;
            using mirror_type = typename view_type::host_mirror_type;

            *field    = 0;
            auto copy = field->deepCopy();
            copy      = copy + 1;

            mirror_type mirrorA = field->getHostMirror();
            mirror_type mirrorB = copy.getHostMirror();

            Kokkos::deep_copy(mirrorA, field->getView());
            Kokkos::deep_copy(mirrorB, copy.getView());

            this->template nestedViewLoop(
                mirrorA, field->getNghost(), [&]<typename... Idx>(const Idx... args) {
                    assertTypeParam<TypeParam>(mirrorA(args...) + 1, mirrorB(args...));
                });
        };

    this->apply(check, this->fields);
}

TYPED_TEST(BareFieldTest, Sum) {
    TypeParam val                           = 1.0;
    TypeParam expected[TestFixture::MaxDim] = {val * this->nPoints[0]};
    for (unsigned d = 1; d < TestFixture::MaxDim; d++) {
        expected[d] = expected[d - 1] * this->nPoints[d];
    }

    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            *field        = val;
            TypeParam sum = field->sum();
            assertTypeParam<TypeParam>(expected[TestFixture::dimToIndex(Dim)], sum);
        };

    this->apply(check, this->fields);
}

TYPED_TEST(BareFieldTest, Min) {
    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            using view_type = typename TestFixture::template field_type<Dim>::view_type;

            const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();
            view_type view                = field->getView();

            Kokkos::parallel_for("Set field", field->getFieldRangePolicy(),
                                 FieldVal<TypeParam, Dim>{view, lDom});
            Kokkos::fence();

            TypeParam min = field->min();
            // minimum value in 3D: -1 + nghost + nghost + nghost
            assertTypeParam<TypeParam>(min, field->getNghost() * Dim - 1);
        };

    this->apply(check, this->fields);
}

TYPED_TEST(BareFieldTest, Max) {
    TypeParam val                           = 1.;
    TypeParam expected[TestFixture::MaxDim] = {this->nPoints[0] - val};
    for (unsigned d = 1; d < TestFixture::MaxDim; d++) {
        expected[d] = expected[d - 1] + this->nPoints[d];
    }
    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            using view_type = typename TestFixture::template field_type<Dim>::view_type;

            const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();
            view_type view                = field->getView();

            Kokkos::parallel_for("Set field", field->getFieldRangePolicy(),
                                 FieldVal<TypeParam, Dim>{view, lDom});
            Kokkos::fence();

            TypeParam max = field->max();
            assertTypeParam<TypeParam>(max, expected[TestFixture::dimToIndex(Dim)]);
        };

    this->apply(check, this->fields);
}

TYPED_TEST(BareFieldTest, Prod) {
    TypeParam sizes[TestFixture::MaxDim] = {(TypeParam)this->nPoints[0]};
    for (unsigned d = 1; d < TestFixture::MaxDim; d++) {
        sizes[d] = sizes[d - 1] * this->nPoints[d];
    }
    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            *field        = 2.;
            TypeParam val = field->prod();

            assertTypeParam<TypeParam>(val, pow(2, sizes[TestFixture::dimToIndex(Dim)]));
        };

    this->apply(check, this->fields);
}

TYPED_TEST(BareFieldTest, ScalarMultiplication) {
    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            using view_type   = typename TestFixture::template field_type<Dim>::view_type;
            using mirror_type = typename view_type::host_mirror_type;

            *field = 1.;
            *field = *field * 10;

            const int shift = field->getNghost();

            view_type view     = field->getView();
            mirror_type mirror = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(mirror, view);

            this->template nestedViewLoop(mirror, shift, [&]<typename... Idx>(const Idx... args) {
                assertTypeParam<TypeParam>(mirror(args...), 10.);
            });
        };

    this->apply(check, this->fields);
}

TYPED_TEST(BareFieldTest, DotProduct) {
    auto check = [&]<unsigned Dim>(
                     std::shared_ptr<typename TestFixture::template field_type<Dim>>& field,
                     std::shared_ptr<typename TestFixture::template vfield_type<Dim>>& vfield) {
        using view_type   = typename TestFixture::template field_type<Dim>::view_type;
        using mirror_type = typename view_type::host_mirror_type;

        *vfield = 1.;
        *field  = 5. * dot(*vfield, *vfield);

        const int shift = field->getNghost();

        view_type view     = field->getView();
        mirror_type mirror = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(mirror, view);

        this->template nestedViewLoop(mirror, shift, [&]<typename... Idx>(const Idx... args) {
            assertTypeParam<TypeParam>(mirror(args...), 5 * Dim);
        });
    };

    this->apply(check, this->fields, this->vfields);
}

TYPED_TEST(BareFieldTest, AllFuncs) {
    auto check = [&]<unsigned Dim>(
                     std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
        using Kokkos::sin, Kokkos::cos, Kokkos::tan, Kokkos::acos, Kokkos::asin, Kokkos::exp,
            Kokkos::erf, Kokkos::cosh, Kokkos::tanh, Kokkos::sinh, Kokkos::log, Kokkos::ceil,
            Kokkos::atan, Kokkos::log, Kokkos::log10, Kokkos::sqrt, Kokkos::floor;
        using view_type   = typename TestFixture::template field_type<Dim>::view_type;
        using mirror_type = typename view_type::host_mirror_type;

        TypeParam pi    = Kokkos::numbers::pi_v<TypeParam>;
        TypeParam alpha = pi / 4;
        *field          = alpha;
        // Compute new value
        TypeParam beta = fabs(
            7.0 * (sin(alpha) * cos(alpha)) / (tan(alpha) * acos(alpha)) - exp(alpha) + erf(alpha)
            + (asin(alpha) * cosh(alpha)) / (atan(alpha) * sinh(alpha)) + tanh(alpha) * log(alpha)
            - log10(alpha) * sqrt(alpha) + floor(alpha) * ceil(alpha));

        // Compute same value via field ops
        *field = fabs(7.0 * (sin(*field) * cos(*field)) / (tan(*field) * acos(*field)) - exp(*field)
                      + erf(*field) + (asin(*field) * cosh(*field)) / (atan(*field) * sinh(*field))
                      + tanh(*field) * log(*field) - log10(*field) * sqrt(*field)
                      + floor(*field) * ceil(*field));

        const int shift = field->getNghost();

        view_type view     = field->getView();
        mirror_type mirror = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(mirror, view);

        this->template nestedViewLoop(mirror, shift, [&]<typename... Idx>(const Idx... args) {
            assertTypeParam<TypeParam>(mirror(args...), beta);
        });
    };

    this->apply(check, this->fields);
}

int main(int argc, char* argv[]) {
    int success = 1;
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
