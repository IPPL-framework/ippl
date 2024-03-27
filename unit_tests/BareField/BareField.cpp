//
// Unit test BareFieldTest
//   Test the functionality of the class BareField.
//
#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstring>

#include "Utility/TypeUtils.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class BareFieldTest;

template <typename T, typename ExecSpace, unsigned Dim>
class BareFieldTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type              = T;
    using exec_space              = ExecSpace;
    constexpr static unsigned dim = Dim;

    using field_type  = ippl::BareField<T, Dim, ExecSpace>;
    using vfield_type = ippl::BareField<ippl::Vector<T, Dim>, Dim, ExecSpace>;

    BareFieldTest()
        : nPoints(getGridSizes<Dim>()) {
        std::array<ippl::Index, Dim> indices;
        for (unsigned d = 0; d < Dim; d++) {
            indices[d] = ippl::Index(nPoints[d]);
        }
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(indices);

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        layout = ippl::FieldLayout<Dim>(MPI_COMM_WORLD, owned, isParallel);

        field  = std::make_shared<field_type>(layout);
        vfield = std::make_shared<vfield_type>(layout);
    }

    ippl::FieldLayout<Dim> layout;

    std::shared_ptr<field_type> field;
    std::shared_ptr<vfield_type> vfield;

    std::array<size_t, Dim> nPoints;
};

template <typename Params>
struct FieldVal {
    constexpr static unsigned Dim = BareFieldTest<Params>::dim;
    using T                       = typename BareFieldTest<Params>::value_type;

    const typename BareFieldTest<Params>::field_type::view_type view;
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
template <typename Params>
struct VFieldVal {
    constexpr static unsigned Dim = BareFieldTest<Params>::dim;
    using T                       = typename BareFieldTest<Params>::value_type;

    const typename BareFieldTest<Params>::vfield_type::view_type view;
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
using Tests = TestParams::tests<1, 2, 3, 4, 5, 6>;
TYPED_TEST_CASE(BareFieldTest, Tests);

TYPED_TEST(BareFieldTest, DeepCopy) {
    auto& field = this->field;

    *field    = 0;
    auto copy = field->deepCopy();
    copy      = copy + 1;

    auto mirrorA = field->getHostMirror();
    auto mirrorB = copy.getHostMirror();

    Kokkos::deep_copy(mirrorA, field->getView());
    Kokkos::deep_copy(mirrorB, copy.getView());

    nestedViewLoop(mirrorA, field->getNghost(), [&]<typename... Idx>(const Idx... args) {
        assertEqual<typename TestFixture::value_type>(mirrorA(args...) + 1, mirrorB(args...));
    });
}

TYPED_TEST(BareFieldTest, Sum) {
    using T = typename TestFixture::value_type;

    T val      = 1.0;
    T expected = std::reduce(this->nPoints.begin(), this->nPoints.end(), val, std::multiplies<>{});

    auto& field = this->field;

    *field = val;
    T sum  = field->sum();
    assertEqual<T>(expected, sum);
    auto& vfield = this->vfield;
    *vfield = ippl::Vector<T, TestFixture::dim>(val);
    ippl::Vector<T, TestFixture::dim> vsum  = vfield->sum();
    for(unsigned d = 0;d < TestFixture::dim;d++){
        assertEqual<T>(expected, vsum[d]);
    }
}

TYPED_TEST(BareFieldTest, Min) {
    using T = typename TestFixture::value_type;

    auto& field = this->field;

    const auto lDom = field->getLayout().getLocalNDIndex();
    auto view       = field->getView();

    Kokkos::parallel_for("Set field", field->getFieldRangePolicy(),
                         FieldVal<TypeParam>{view, lDom});
    auto& vfield = this->vfield;
    auto  vview       = vfield->getView();
    Kokkos::parallel_for("Set field", field->getFieldRangePolicy(),
                         VFieldVal<TypeParam>{vview, lDom});
    Kokkos::fence();

    auto min = field->min();
    // minimum value in 3D: -1 + nghost + nghost + nghost
    assertEqual<typename TestFixture::value_type>(min, field->getNghost() * TestFixture::dim - 1);

    ippl::Vector<T, TestFixture::dim> vsum  = vfield->min();
    for(unsigned d = 0;d < TestFixture::dim;d++){
        assertEqual<T>(field->getNghost() * TestFixture::dim - 1, vsum[d]);
    }
}

TYPED_TEST(BareFieldTest, Max) {
    using T    = typename TestFixture::value_type;
    T val      = 1.;
    T expected = std::accumulate(this->nPoints.begin(), this->nPoints.end(), -val);

    auto& field = this->field;

    const auto lDom = field->getLayout().getLocalNDIndex();
    auto view       = field->getView();

    Kokkos::parallel_for("Set field", field->getFieldRangePolicy(),
                         FieldVal<TypeParam>{view, lDom});
    auto& vfield = this->vfield;
    auto  vview       = vfield->getView();
    Kokkos::parallel_for("Set field", field->getFieldRangePolicy(),
                         VFieldVal<TypeParam>{vview, lDom});
    Kokkos::fence();

    T max = field->max();
    assertEqual<T>(max, expected);

    ippl::Vector<T, TestFixture::dim> vsum  = vfield->max();
    for(unsigned d = 0;d < TestFixture::dim;d++){
        assertEqual<T>(expected, vsum[d]);
    }
}

TYPED_TEST(BareFieldTest, Prod) {
    using T = typename TestFixture::value_type;
    T size  = std::reduce(this->nPoints.begin(), this->nPoints.end(), 1, std::multiplies<>{});

    auto& field = this->field;

    *field = 2.;
    T val  = field->prod();

    assertEqual<T>(val, pow(2, size));
}

TYPED_TEST(BareFieldTest, ScalarMultiplication) {
    auto& field = this->field;

    *field = 1.;
    *field = *field * 10;

    const int shift = field->getNghost();

    auto view   = field->getView();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    nestedViewLoop(mirror, shift, [&]<typename... Idx>(const Idx... args) {
        assertEqual<typename TestFixture::value_type>(mirror(args...), 10.);
    });
}

TYPED_TEST(BareFieldTest, DotProduct) {
    auto& field  = this->field;
    auto& vfield = this->vfield;

    *vfield = 1.;
    *field  = 5. * dot(*vfield, *vfield);

    const int shift = field->getNghost();

    auto view   = field->getView();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    nestedViewLoop(mirror, shift, [&]<typename... Idx>(const Idx... args) {
        assertEqual<typename TestFixture::value_type>(mirror(args...), 5 * TestFixture::dim);
    });
}

TYPED_TEST(BareFieldTest, AllFuncs) {
    using T = typename TestFixture::value_type;
    using Kokkos::sin, Kokkos::cos, Kokkos::tan, Kokkos::acos, Kokkos::asin, Kokkos::exp,
        Kokkos::erf, Kokkos::cosh, Kokkos::tanh, Kokkos::sinh, Kokkos::log, Kokkos::ceil,
        Kokkos::atan, Kokkos::log, Kokkos::log10, Kokkos::sqrt, Kokkos::floor;

    auto& field = this->field;

    T pi    = Kokkos::numbers::pi_v<T>;
    T alpha = pi / 4;
    *field  = alpha;
    // Compute new value
    T beta =
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

    nestedViewLoop(mirror, shift, [&]<typename... Idx>(const Idx... args) {
        assertEqual<T>(mirror(args...), beta);
    });
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
