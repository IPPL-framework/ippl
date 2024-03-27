//
// Unit test UniformCartesianTest
//   Test functionality of the class UniformCartesian.
//
#include "Ippl.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class UniformCartesianTest;

template <typename T, unsigned Dim>
class UniformCartesianTest<Parameters<T, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type              = T;
    constexpr static unsigned dim = Dim;

    UniformCartesianTest()
        : nPoints(getGridSizes<Dim>()) {}

    ippl::NDIndex<Dim> createMesh(ippl::Vector<T, Dim>& hx, ippl::Vector<T, Dim>& origin,
                                  T& cellVol, T& meshVol) {
        std::array<ippl::Index, Dim> args;
        for (unsigned d = 0; d < Dim; d++) {
            args[d] = ippl::Index(nPoints[d]);
        }
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        cellVol = 1;
        meshVol = 1;
        for (unsigned d = 0; d < Dim; d++) {
            hx[d] = (d + 1.) / nPoints[d];
            meshVol *= d + 1;
            cellVol *= hx[d];
            origin[d] = 0;
        }

        return owned;
    }

    std::array<size_t, Dim> nPoints;
};

using Precisions = TestParams::Precisions;
using Ranks      = TestParams::Ranks<1, 2, 3, 4, 5, 6>;
using Tests      = TestForTypes<CreateCombinations<Precisions, Ranks>::type>::type;
TYPED_TEST_CASE(UniformCartesianTest, Tests);

TYPED_TEST(UniformCartesianTest, Constructor) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    ippl::Vector<T, Dim> hx;
    ippl::Vector<T, Dim> origin;
    T cellVol, meshVol;

    ippl::NDIndex<Dim> owned = this->createMesh(hx, origin, cellVol, meshVol);
    ippl::UniformCartesian<T, Dim> mesh(owned, hx, origin);

    T length = mesh.getCellVolume();

    assertEqual(length, cellVol);
    assertEqual(mesh.getMeshVolume(), meshVol);
}

TYPED_TEST(UniformCartesianTest, Initialize) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    ippl::Vector<T, Dim> hx;
    ippl::Vector<T, Dim> origin;
    T cellVol, meshVol;

    ippl::NDIndex<Dim> owned = this->createMesh(hx, origin, cellVol, meshVol);

    ippl::UniformCartesian<T, Dim> mesh;
    mesh.initialize(owned, hx, origin);

    assertEqual(mesh.getCellVolume(), cellVol);
    assertEqual(mesh.getMeshVolume(), meshVol);
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
