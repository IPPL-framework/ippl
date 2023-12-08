//
// Unit test ParameterListTest
//   Test functionality of the class ParameterList.
//
#include "Ippl.h"

#include "Utility/ParameterList.h"

#include "Utility/IpplException.h"

#include "gtest/gtest.h"

class ParameterListTest : public ::testing::Test {
public:
    ParameterListTest() {}
};

TEST_F(ParameterListTest, Add) {
    ippl::ParameterList p;
    p.add<double>("tolerance", 1.0e-8);

    bool isContained = false;
    try {
        p.add<double>("tolerance", 1.0e-9);
    } catch (...) {
        isContained = true;
    }

    ASSERT_TRUE(isContained);
}

TEST_F(ParameterListTest, UpdateSingle) {
    double tol = 1.0e-6;

    ippl::ParameterList p;
    p.add<double>("tolerance", 1.0e-9);

    p.update("tolerance", tol);

    ASSERT_DOUBLE_EQ(p.get<double>("tolerance"), tol);

    bool isContained = true;
    try {
        p.update<bool>("enable", true);
    } catch (...) {
        isContained = false;
    }

    ASSERT_FALSE(isContained);
}

TEST_F(ParameterListTest, Merge) {
    ippl::ParameterList p1;
    p1.add<double>("tolerance", 1.0e-8);
    p1.add<bool>("is enabled", false);

    ippl::ParameterList p2;

    double tol = 1.0e-12;
    int size   = 5;

    p2.add<int>("size", size);
    p2.add<double>("tolerance", tol);

    p1.merge(p2);

    ASSERT_DOUBLE_EQ(p1.get<double>("tolerance"), tol);
    ASSERT_EQ(p1.get<int>("size"), size);
    ASSERT_FALSE(p1.get<bool>("is enabled"));
}

TEST_F(ParameterListTest, Update) {
    ippl::ParameterList p1;
    p1.add<double>("tolerance", 1.0e-8);
    p1.add<bool>("is enabled", false);

    ippl::ParameterList p2;

    double tol = 1.0e-12;

    p2.add<int>("size", 5);
    p2.add<double>("tolerance", tol);

    p1.update(p2);

    // Update only modifies the values of the existing
    // parameters, does not add new parameters to p1,
    // so "size" should not be contained in p1
    // we try to get "size" from p1, which should not
    // run, so isContained should continue being false.

    bool isContained = false;
    try {
        isContained = p1.get<int>("size") == 5;
    } catch (...) {
        // do nothing here
    }

    ASSERT_DOUBLE_EQ(p1.get<double>("tolerance"), tol);
    ASSERT_FALSE(isContained);
    ASSERT_FALSE(p1.get<bool>("is enabled"));
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
