

#include "Ippl.h"

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class LagrangeSpaceTest;

template <typename T, unsigned Dim>
class LagrangeSpaceTest<Parameters<T, Rank<Dim>>> : public ::testing::Test {};

int main(int argc, char* argv[]) {
    int success = 1;
    TestParams::checkArgs(argc, argv);
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}