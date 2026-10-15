#include "Ippl.h"

#include "gtest/gtest.h"

void assignParticleAttribFromUnusedTranslationUnit(
    ippl::ParticleAttrib<double>& scalar, ippl::ParticleAttrib<ippl::Vector<double, 3>>& vector);

TEST(ParticleAttribAssignment, AssignsOnlyLiveParticlesAcrossTranslationUnits) {
    using Vector            = ippl::Vector<double, 3>;
    using Size              = ippl::detail::size_type;
    constexpr Size Capacity = 8;
    Size particleCount      = Capacity;
    ippl::ParticleAttrib<double> scalar;
    ippl::ParticleAttrib<Vector> vector;
    scalar.setParticleCount(particleCount);
    vector.setParticleCount(particleCount);
    scalar.resize(Capacity);
    vector.resize(Capacity);
    auto scalarStorage = scalar.getView();
    auto vectorStorage = vector.getView();

    // Emit the same assignment specializations in another translation unit without initializing
    // its NVCC lambda helpers. Volatile keeps that caller reachable under link-time optimization.
    volatile bool runUnusedPath = false;
    if (runUnusedPath) {
        assignParticleAttribFromUnusedTranslationUnit(scalar, vector);
    }

    const double scalarValue = 3.25;
    const Vector vectorValue(1.0, -2.0, 3.0);
    for (const Size activeCount : {Size(0), Size(3), Capacity}) {
        SCOPED_TRACE(activeCount);
        Kokkos::deep_copy(scalarStorage, -7.0);
        Kokkos::deep_copy(vectorStorage, Vector(-9.0));
        particleCount = activeCount;

        EXPECT_EQ(&(scalar = scalarValue), &scalar);
        EXPECT_EQ(&(vector = vectorValue), &vector);
        EXPECT_EQ(particleCount, activeCount);
        EXPECT_EQ(scalar.size(), Capacity);
        EXPECT_EQ(vector.size(), Capacity);

        auto scalarHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scalarStorage);
        auto vectorHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vectorStorage);
        for (Size i = 0; i < Capacity; ++i) {
            EXPECT_DOUBLE_EQ(scalarHost(i), i < activeCount ? scalarValue : -7.0);
            for (unsigned d = 0; d < 3; ++d) {
                EXPECT_DOUBLE_EQ(vectorHost(i)[d], i < activeCount ? vectorValue[d] : -9.0);
            }
        }
    }
    EXPECT_EQ(scalar.getView().data(), scalarStorage.data());
    EXPECT_EQ(vector.getView().data(), vectorStorage.data());
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
