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
TYPED_TEST_SUITE(BareFieldTest, Tests);

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

TYPED_TEST(BareFieldTest, IndexedSubdomainScalarAssignment) {
    using T = typename TestFixture::value_type;

    auto& field = this->field;
    *field      = 0;

    ippl::NDIndex<TestFixture::dim> subdomain = field->getDomain();
    subdomain[0] =
        ippl::Index(subdomain[0].first(), subdomain[0].last(), 2 * subdomain[0].stride());

    (*field)[subdomain] = T(7);

    auto mirror = field->getHostMirror();
    Kokkos::deep_copy(mirror, field->getView());

    const auto owned = field->getOwned();
    const int nghost = field->getNghost();
    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        std::array<ippl::Index, TestFixture::dim> pointIndices;
        const std::array<size_t, TestFixture::dim> viewCoords{static_cast<size_t>(args)...};
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            const int global = owned[d].first()
                               + static_cast<int>(viewCoords[d] - nghost) * owned[d].stride();
            pointIndices[d] = ippl::Index(global, global);
        }
        auto point = std::make_from_tuple<ippl::NDIndex<TestFixture::dim>>(pointIndices);
        const T expected = subdomain.contains(point) ? T(7) : T(0);
        assertEqual<T>(expected, mirror(args...));
    });
}

TYPED_TEST(BareFieldTest, IndexedSubdomainExpressionAssignment) {
    using T = typename TestFixture::value_type;

    auto& lhs = this->field;
    lhs->operator=(0);

    typename TestFixture::field_type rhs(this->layout);
    rhs = T(5);

    ippl::NDIndex<TestFixture::dim> subdomain = lhs->getDomain();
    subdomain[0] =
        ippl::Index(subdomain[0].first(), subdomain[0].last(), 2 * subdomain[0].stride());

    (*lhs)[subdomain] = rhs[subdomain] + T(2);

    auto mirror = lhs->getHostMirror();
    Kokkos::deep_copy(mirror, lhs->getView());

    const auto owned = lhs->getOwned();
    const int nghost = lhs->getNghost();
    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        std::array<ippl::Index, TestFixture::dim> pointIndices;
        const std::array<size_t, TestFixture::dim> viewCoords{static_cast<size_t>(args)...};
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            const int global = owned[d].first()
                               + static_cast<int>(viewCoords[d] - nghost) * owned[d].stride();
            pointIndices[d] = ippl::Index(global, global);
        }
        auto point = std::make_from_tuple<ippl::NDIndex<TestFixture::dim>>(pointIndices);
        const T expected = subdomain.contains(point) ? T(7) : T(0);
        assertEqual<T>(expected, mirror(args...));
    });
}

TYPED_TEST(BareFieldTest, ChainedIndexedSubdomainScalarAssignment) {
    using T = typename TestFixture::value_type;

    auto& field = this->field;
    *field      = 0;

    ippl::NDIndex<TestFixture::dim> subdomain = field->getDomain();
    subdomain[0] =
        ippl::Index(subdomain[0].first(), subdomain[0].last(), 2 * subdomain[0].stride());

    if constexpr (TestFixture::dim == 1) {
        (*field)[subdomain[0]] = T(7);
    } else if constexpr (TestFixture::dim == 2) {
        (*field)[subdomain[0]][subdomain[1]] = T(7);
    } else if constexpr (TestFixture::dim == 3) {
        (*field)[subdomain[0]][subdomain[1]][subdomain[2]] = T(7);
    } else if constexpr (TestFixture::dim == 4) {
        (*field)[subdomain[0]][subdomain[1]][subdomain[2]][subdomain[3]] = T(7);
    } else if constexpr (TestFixture::dim == 5) {
        (*field)[subdomain[0]][subdomain[1]][subdomain[2]][subdomain[3]][subdomain[4]] = T(7);
    } else if constexpr (TestFixture::dim == 6) {
        (*field)[subdomain[0]][subdomain[1]][subdomain[2]][subdomain[3]][subdomain[4]]
                [subdomain[5]] = T(7);
    }

    auto mirror = field->getHostMirror();
    Kokkos::deep_copy(mirror, field->getView());

    const auto owned = field->getOwned();
    const int nghost = field->getNghost();
    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        std::array<ippl::Index, TestFixture::dim> pointIndices;
        const std::array<size_t, TestFixture::dim> viewCoords{static_cast<size_t>(args)...};
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            const int global = owned[d].first()
                               + static_cast<int>(viewCoords[d] - nghost) * owned[d].stride();
            pointIndices[d] = ippl::Index(global, global);
        }
        auto point = std::make_from_tuple<ippl::NDIndex<TestFixture::dim>>(pointIndices);
        const T expected = subdomain.contains(point) ? T(7) : T(0);
        assertEqual<T>(expected, mirror(args...));
    });
}

TYPED_TEST(BareFieldTest, ChainedIndexedSubdomainExpressionAssignment) {
    using T = typename TestFixture::value_type;

    auto& lhs = this->field;
    lhs->operator=(0);

    typename TestFixture::field_type rhs(this->layout);
    rhs = T(5);

    ippl::NDIndex<TestFixture::dim> subdomain = lhs->getDomain();
    subdomain[0] =
        ippl::Index(subdomain[0].first(), subdomain[0].last(), 2 * subdomain[0].stride());

    if constexpr (TestFixture::dim == 1) {
        (*lhs)[subdomain[0]] = rhs[subdomain[0]] + T(2);
    } else if constexpr (TestFixture::dim == 2) {
        (*lhs)[subdomain[0]][subdomain[1]] = rhs[subdomain[0]][subdomain[1]] + T(2);
    } else if constexpr (TestFixture::dim == 3) {
        (*lhs)[subdomain[0]][subdomain[1]][subdomain[2]] =
            rhs[subdomain[0]][subdomain[1]][subdomain[2]] + T(2);
    } else if constexpr (TestFixture::dim == 4) {
        (*lhs)[subdomain[0]][subdomain[1]][subdomain[2]][subdomain[3]] =
            rhs[subdomain[0]][subdomain[1]][subdomain[2]][subdomain[3]] + T(2);
    } else if constexpr (TestFixture::dim == 5) {
        (*lhs)[subdomain[0]][subdomain[1]][subdomain[2]][subdomain[3]][subdomain[4]] =
            rhs[subdomain[0]][subdomain[1]][subdomain[2]][subdomain[3]][subdomain[4]] + T(2);
    } else if constexpr (TestFixture::dim == 6) {
        (*lhs)[subdomain[0]][subdomain[1]][subdomain[2]][subdomain[3]][subdomain[4]]
              [subdomain[5]] =
                  rhs[subdomain[0]][subdomain[1]][subdomain[2]][subdomain[3]][subdomain[4]]
                     [subdomain[5]]
                  + T(2);
    }

    auto mirror = lhs->getHostMirror();
    Kokkos::deep_copy(mirror, lhs->getView());

    const auto owned = lhs->getOwned();
    const int nghost = lhs->getNghost();
    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        std::array<ippl::Index, TestFixture::dim> pointIndices;
        const std::array<size_t, TestFixture::dim> viewCoords{static_cast<size_t>(args)...};
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            const int global = owned[d].first()
                               + static_cast<int>(viewCoords[d] - nghost) * owned[d].stride();
            pointIndices[d] = ippl::Index(global, global);
        }
        auto point = std::make_from_tuple<ippl::NDIndex<TestFixture::dim>>(pointIndices);
        const T expected = subdomain.contains(point) ? T(7) : T(0);
        assertEqual<T>(expected, mirror(args...));
    });
}

TYPED_TEST(BareFieldTest, SparseIndexedSubdomainScalarAssignment) {
    using T = typename TestFixture::value_type;

    auto& field = this->field;
    *field      = 0;

    ippl::SIndex<TestFixture::dim> sindex(this->layout);
    const auto domain = field->getDomain();

    for (int i = domain[0].first(); i <= domain[0].last(); i += 2 * domain[0].stride()) {
        typename ippl::SIndex<TestFixture::dim>::point_type point;
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            point[d] = domain[d].first();
        }
        point[0] = i;
        sindex.addIndex(point);
    }

    (*field)[sindex] = T(7);

    auto mirror = field->getHostMirror();
    Kokkos::deep_copy(mirror, field->getView());

    const auto owned = field->getOwned();
    const int nghost = field->getNghost();
    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        typename ippl::SIndex<TestFixture::dim>::point_type point;
        const std::array<size_t, TestFixture::dim> viewCoords{static_cast<size_t>(args)...};
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            point[d] = owned[d].first()
                       + static_cast<int>(viewCoords[d] - nghost) * owned[d].stride();
        }
        const T expected = sindex.hasIndex(point) ? T(7) : T(0);
        assertEqual<T>(expected, mirror(args...));
    });
}

TYPED_TEST(BareFieldTest, SparseIndexedSubdomainExpressionAssignment) {
    using T = typename TestFixture::value_type;

    auto& lhs = this->field;
    lhs->operator=(0);

    typename TestFixture::field_type rhs(this->layout);
    rhs = T(5);

    ippl::SIndex<TestFixture::dim> sindex(this->layout);
    const auto domain = lhs->getDomain();

    for (int i = domain[0].first(); i <= domain[0].last(); i += 2 * domain[0].stride()) {
        typename ippl::SIndex<TestFixture::dim>::point_type point;
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            point[d] = domain[d].first();
        }
        point[0] = i;
        sindex.addIndex(point);
    }

    (*lhs)[sindex] = rhs[sindex] + T(2);

    auto mirror = lhs->getHostMirror();
    Kokkos::deep_copy(mirror, lhs->getView());

    const auto owned = lhs->getOwned();
    const int nghost = lhs->getNghost();
    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        typename ippl::SIndex<TestFixture::dim>::point_type point;
        const std::array<size_t, TestFixture::dim> viewCoords{static_cast<size_t>(args)...};
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            point[d] = owned[d].first()
                       + static_cast<int>(viewCoords[d] - nghost) * owned[d].stride();
        }
        const T expected = sindex.hasIndex(point) ? T(7) : T(0);
        assertEqual<T>(expected, mirror(args...));
    });
}

TYPED_TEST(BareFieldTest, SparseIndexedOffsetExpressionAssignment) {
    using T = typename TestFixture::value_type;

    auto& lhs = this->field;
    lhs->operator=(0);

    typename TestFixture::field_type rhs(this->layout);
    rhs = T(5);
    rhs.fillHalo();

    ippl::SIndex<TestFixture::dim> sindex(this->layout);
    const auto domain = lhs->getDomain();
    const int offsetValue = domain[0].stride();

    for (int i = domain[0].first(); i <= domain[0].last() - offsetValue;
         i += 2 * domain[0].stride()) {
        typename ippl::SIndex<TestFixture::dim>::point_type point;
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            point[d] = domain[d].first();
        }
        point[0] = i;
        sindex.addIndex(point);
    }

    ippl::SOffset<TestFixture::dim> offset;
    offset[0] = offsetValue;

    (*lhs)[sindex] = rhs[sindex + offset] + T(2);

    auto mirror = lhs->getHostMirror();
    Kokkos::deep_copy(mirror, lhs->getView());

    const auto owned = lhs->getOwned();
    const int nghost = lhs->getNghost();
    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        typename ippl::SIndex<TestFixture::dim>::point_type point;
        const std::array<size_t, TestFixture::dim> viewCoords{static_cast<size_t>(args)...};
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            point[d] = owned[d].first()
                       + static_cast<int>(viewCoords[d] - nghost) * owned[d].stride();
        }
        const T expected = sindex.hasIndex(point) ? T(7) : T(0);
        assertEqual<T>(expected, mirror(args...));
    });
}

TYPED_TEST(BareFieldTest, SparseIndexedCallOffsetExpressionAssignment) {
    using T = typename TestFixture::value_type;

    auto& lhs = this->field;
    lhs->operator=(0);

    typename TestFixture::field_type rhs(this->layout);
    rhs = T(5);
    rhs.fillHalo();

    ippl::SIndex<TestFixture::dim> sindex(this->layout);
    const auto domain = lhs->getDomain();
    const int offsetValue = domain[0].stride();

    for (int i = domain[0].first(); i <= domain[0].last() - offsetValue;
         i += 2 * domain[0].stride()) {
        typename ippl::SIndex<TestFixture::dim>::point_type point;
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            point[d] = domain[d].first();
        }
        point[0] = i;
        sindex.addIndex(point);
    }

    ippl::SOffset<TestFixture::dim> offset;
    offset[0] = offsetValue;

    (*lhs)[sindex] = rhs[sindex(offset)] + T(2);

    auto mirror = lhs->getHostMirror();
    Kokkos::deep_copy(mirror, lhs->getView());

    const auto owned = lhs->getOwned();
    const int nghost = lhs->getNghost();
    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        typename ippl::SIndex<TestFixture::dim>::point_type point;
        const std::array<size_t, TestFixture::dim> viewCoords{static_cast<size_t>(args)...};
        for (unsigned d = 0; d < TestFixture::dim; ++d) {
            point[d] = owned[d].first()
                       + static_cast<int>(viewCoords[d] - nghost) * owned[d].stride();
        }
        const T expected = sindex.hasIndex(point) ? T(7) : T(0);
        assertEqual<T>(expected, mirror(args...));
    });
}

TYPED_TEST(BareFieldTest, SparseIndexSetOperations) {
    ippl::SIndex<TestFixture::dim> sindex(this->layout);
    const auto domain = this->field->getDomain();

    typename ippl::SIndex<TestFixture::dim>::point_type firstPoint;
    typename ippl::SIndex<TestFixture::dim>::point_type secondPoint;
    for (unsigned d = 0; d < TestFixture::dim; ++d) {
        firstPoint[d]  = domain[d].first();
        secondPoint[d] = domain[d].first();
    }
    secondPoint[0] = domain[0].first() + domain[0].stride();

    EXPECT_TRUE(sindex.addIndex(firstPoint));
    EXPECT_TRUE(sindex.addIndex(secondPoint));
    EXPECT_FALSE(sindex.addIndex(firstPoint));

    ippl::NDIndex<TestFixture::dim> subset = domain;
    subset[0] = ippl::Index(firstPoint[0], firstPoint[0], domain[0].stride());
    sindex &= subset;

    EXPECT_TRUE(sindex.hasIndex(firstPoint));
    EXPECT_FALSE(sindex.hasIndex(secondPoint));

    ippl::SOffset<TestFixture::dim> secondOffset;
    for (unsigned d = 0; d < TestFixture::dim; ++d) {
        secondOffset[d] = secondPoint[d];
    }
    sindex |= secondOffset;

    EXPECT_TRUE(sindex.hasIndex(secondPoint));
    EXPECT_EQ(sindex.size(), 2u);
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
