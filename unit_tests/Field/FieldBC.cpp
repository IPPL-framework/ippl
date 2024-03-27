//
// Unit test FieldBC
//   Test field boundary conditions.
//
//
#include "Ippl.h"

#include "Utility/IpplException.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class FieldBCTest;

template <typename T, typename ExecSpace, unsigned Dim>
class FieldBCTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type              = T;
    using exec_space              = ExecSpace;
    constexpr static unsigned dim = Dim;

    using mesh_type      = ippl::UniformCartesian<T, Dim>;
    using centering_type = typename mesh_type::DefaultCentering;
    using field_type     = ippl::Field<T, Dim, mesh_type, centering_type, ExecSpace>;
    using bc_type        = ippl::BConds<field_type, Dim>;

    FieldBCTest()
        : nPoints(getGridSizes<Dim>()) {
        for (unsigned d = 0; d < Dim; d++) {
            domain[d] = nPoints[d] / 10;
        }

        std::array<ippl::Index, Dim> indices;
        for (unsigned d = 0; d < Dim; d++) {
            indices[d] = ippl::Index(nPoints[d]);
        }
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(indices);

        ippl::Vector<T, Dim> hx;
        ippl::Vector<T, Dim> origin;

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);  // Specifies SERIAL, PARALLEL dims

        for (unsigned int d = 0; d < Dim; d++) {
            hx[d]     = domain[d] / nPoints[d];
            origin[d] = 0;
        }

        layout = ippl::FieldLayout<Dim>(MPI_COMM_WORLD, owned, isParallel);
        mesh   = mesh_type(owned, hx, origin);

        field  = std::make_shared<field_type>(mesh, layout);
        *field = 1.0;
        *field = (*field) * 10.0;
        HostF  = field->getHostMirror();
    }

    void checkResult(const T expected) {
        const auto& lDomains = layout.getHostLocalDomains();
        const auto& domain   = layout.getDomain();
        const int myRank     = ippl::Comm->rank();

        Kokkos::deep_copy(HostF, field->getView());

        for (size_t face = 0; face < 2UL * Dim; ++face) {
            size_t d        = face / 2;
            bool checkUpper = lDomains[myRank][d].max() == domain[d].max();
            bool checkLower = lDomains[myRank][d].min() == domain[d].min();
            if (!checkUpper && !checkLower) {
                continue;
            }
            int N = HostF.extent(d);
            nestedLoop<Dim>(
                [&](unsigned) {
                    return 1;
                },
                [&](unsigned dim) {
                    return dim == d ? 2 : HostF.extent(dim) - 1;
                },
                [&]<typename... Idx>(const Idx... args) {
                    using index_type = std::tuple_element_t<0, std::tuple<Idx...>>;

                    index_type coords[Dim] = {args...};
                    if (checkLower) {
                        coords[d] = 0;
                        EXPECT_DOUBLE_EQ(expected, ippl::apply(HostF, coords));
                    }
                    if (checkUpper) {
                        coords[d] = N - 1;
                        EXPECT_DOUBLE_EQ(expected, ippl::apply(HostF, coords));
                    }
                });
        }
    }

    ippl::FieldLayout<Dim> layout;
    std::shared_ptr<field_type> field;
    bc_type bcField;

    mesh_type mesh;

    using mirror_type = typename field_type::view_type::host_mirror_type;
    mirror_type HostF;

    std::array<size_t, Dim> nPoints;
    std::array<T, Dim> domain;
};

using Tests = TestParams::tests<1, 2, 3, 4, 5, 6>;
TYPED_TEST_CASE(FieldBCTest, Tests);

TYPED_TEST(FieldBCTest, PeriodicBC) {
    using T    = typename TestFixture::value_type;
    T expected = 10.0;

    auto& field   = this->field;
    auto& bcField = this->bcField;

    for (size_t i = 0; i < 2 * TestFixture::dim; ++i) {
        bcField[i] = std::make_shared<ippl::PeriodicFace<typename TestFixture::field_type>>(i);
    }
    bcField.findBCNeighbors(*field);
    bcField.apply(*field);
    this->checkResult(expected);
}

TYPED_TEST(FieldBCTest, NoBC) {
    using T    = typename TestFixture::value_type;
    T expected = 1.0;

    auto& field   = this->field;
    auto& bcField = this->bcField;

    for (size_t i = 0; i < 2 * TestFixture::dim; ++i) {
        bcField[i] = std::make_shared<ippl::NoBcFace<typename TestFixture::field_type>>(i);
    }
    bcField.findBCNeighbors(*field);
    bcField.apply(*field);
    this->checkResult(expected);
}

TYPED_TEST(FieldBCTest, ZeroBC) {
    using T    = typename TestFixture::value_type;
    T expected = 0.0;

    auto& field   = this->field;
    auto& bcField = this->bcField;

    for (size_t i = 0; i < 2 * TestFixture::dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<typename TestFixture::field_type>>(i);
    }
    bcField.findBCNeighbors(*field);
    bcField.apply(*field);
    this->checkResult(expected);
}

TYPED_TEST(FieldBCTest, ConstantBC) {
    using T    = typename TestFixture::value_type;
    T constant = 7.0;

    auto& field   = this->field;
    auto& bcField = this->bcField;

    for (size_t i = 0; i < 2 * TestFixture::dim; ++i) {
        bcField[i] =
            std::make_shared<ippl::ConstantFace<typename TestFixture::field_type>>(i, constant);
    }
    bcField.findBCNeighbors(*field);
    bcField.apply(*field);
    this->checkResult(constant);
}

TYPED_TEST(FieldBCTest, ExtrapolateBC) {
    using T    = typename TestFixture::value_type;
    T expected = 10.0;

    auto& field   = this->field;
    auto& bcField = this->bcField;

    for (size_t i = 0; i < 2 * TestFixture::dim; ++i) {
        bcField[i] =
            std::make_shared<ippl::ExtrapolateFace<typename TestFixture::field_type>>(i, 0.0, 1.0);
    }
    bcField.findBCNeighbors(*field);
    bcField.apply(*field);
    this->checkResult(expected);
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
