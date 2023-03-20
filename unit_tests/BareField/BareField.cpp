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

#include <array>
#include <cmath>
#include <tuple>

#include "gtest/gtest.h"

template <unsigned... Dims>
class MultirankUtils {
    // Checking for specialization
    // https://stackoverflow.com/a/28796458
    template <typename, template <typename...> class>
    struct is_specialization : std::false_type {};

    template <template <typename...> class Ref, typename... Args>
    struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

    template <typename Functor, typename... Args, unsigned long... Idx>
    void apply_impl(Functor& f, std::tuple<Args...>& args, const std::index_sequence<Idx...>&) {
        if constexpr (is_specialization<std::tuple_element_t<0, std::tuple<Args...>>,
                                        std::tuple>::value) {
            (std::apply(f, std::get<Idx>(args)), ...);
        } else {
            (f(std::get<Idx>(args)), ...);
        }
    }

    template <typename Tester, size_t... Idx>
    void setup_impl(Tester&& t, const std::index_sequence<Idx...>&) {
        ((t->template setupDim<Idx, Dims>()), ...);
    }

    // Tuple zipping
    // https://stackoverflow.com/a/47127033
    template <size_t Idx, typename... Tuples>
    using zipped_element = std::tuple<std::tuple_element_t<Idx, std::decay_t<Tuples>>...>;

    template <size_t Idx, typename... Tuples>
    zipped_element<Idx, Tuples...> zip_at(Tuples&&... ts) {
        return {std::get<Idx>(std::forward<Tuples>(ts))...};
    }

    template <typename... Tuples, size_t... Idx>
    std::tuple<zipped_element<Idx, Tuples...>...> zip_impl(Tuples&&... ts,
                                                           const std::index_sequence<Idx...>&) {
        return {zip_at<Idx>(std::forward<Tuples>(ts)...)...};
    }

protected:
    template <template <unsigned Dim> class Type>
    using Collection = std::tuple<Type<Dims>...>;

    template <template <typename> class Pointer, template <unsigned Dim> class Type>
    using PtrCollection = std::tuple<Pointer<Type<Dims>>...>;

public:
    template <typename Tester>
    void setup(Tester&& t) {
        setup_impl(t, std::make_index_sequence<sizeof...(Dims)>{});
    }

    template <typename Functor, typename... Args>
    auto apply(Functor& f, std::tuple<Args...>& args) {
        return apply_impl(f, args, std::make_index_sequence<sizeof...(Dims)>{});
    }

    template <typename Tuple, typename... Tuples>
    auto zip(Tuple&& t0, Tuples&&... ts) {
        constexpr size_t size = std::tuple_size_v<std::decay_t<Tuple>>;
        static_assert(((std::tuple_size_v<std::decay_t<Tuples>> == size) && ...),
                      "Mismatched tuple sizes");

        return zip_impl<Tuple, Tuples...>(std::forward<Tuple>(t0), std::forward<Tuples>(ts)...,
                                          std::make_index_sequence<size>{});
    }

    // https://stackoverflow.com/questions/34535795/n-dimensionally-nested-metaloops-with-templates
    template <unsigned Dim, class BeginFunctor, class EndFunctor, class Functor>
    constexpr void nestedLoop(BeginFunctor&& begin, EndFunctor&& end, Functor&& c) {
        for (size_t i = begin(Dim); i < end(Dim); ++i) {
            if constexpr (Dim == 1) {
                c(i);
            } else {
                auto next = [i, &c](auto... args) {
                    c(i, args...);
                };
                nestedLoop<Dim - 1>(begin, end, next);
            }
        }
    }
};

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

        Kokkos::parallel_for(
            "Set field", field->getRangePolicy(),
            KOKKOS_LAMBDA<typename... Idx>(const Idx... args) {
                double tot = (args + ...);
                for (unsigned d = 0; d < Dim; d++)
                    tot += lDom[d].first();
                view(args...) = tot - 1;
            });
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

        Kokkos::parallel_for(
            "Set field", field->getRangePolicy(),
            KOKKOS_LAMBDA<typename... Idx>(const Idx... args) {
                double tot = (args + ...);
                for (unsigned d = 0; d < Dim; d++)
                    tot += lDom[d].first();
                view(args...) = tot - 1;
            });
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

        nestedLoop<Dim>(
            [&](unsigned) {
                return shift;
            },
            [&](unsigned d) {
                return mirror.extent(d) - shift;
            },
            [&]<typename... Idx>(const Idx... args) {
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

        nestedLoop<Dim>(
            [&](unsigned) {
                return shift;
            },
            [&](unsigned d) {
                return mirror.extent(d) - shift;
            },
            [&]<typename... Idx>(const Idx... args) {
                ASSERT_DOUBLE_EQ(mirror(args...), 15.);
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

        nestedLoop<Dim>(
            [&](unsigned) {
                return shift;
            },
            [&](unsigned d) {
                return mirror.extent(d) - shift;
            },
            [&]<typename... Idx>(const Idx... args) {
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
