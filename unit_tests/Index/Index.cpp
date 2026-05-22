//
// Unit test Index
//   Test strided Index semantics.
//
#include "Ippl.h"

#include "gtest/gtest.h"

TEST(IndexTest, StridedTouchesRequiresCommonElement) {
    ippl::Index even(0, 10, 2);
    ippl::Index odd(1, 9, 2);

    EXPECT_FALSE(even.touches(odd));
    EXPECT_FALSE(odd.touches(even));
    EXPECT_TRUE(even.intersect(odd).empty());
}

TEST(IndexTest, StridedContainsRequiresContainedElements) {
    ippl::Index even(0, 10, 2);
    ippl::Index odd(1, 9, 2);
    ippl::Index everyFourth(0, 8, 4);

    EXPECT_FALSE(even.contains(odd));
    EXPECT_TRUE(even.contains(everyFourth));
    EXPECT_FALSE(everyFourth.contains(even));
}

TEST(IndexTest, GrowUsesIndexStrideForCoordinateBounds) {
    ippl::Index even(2, 10, 2);
    ippl::Index grown = even.grow(2);

    EXPECT_EQ(grown.first(), -2);
    EXPECT_EQ(grown.last(), 14);
    EXPECT_EQ(grown.stride(), 2);
    EXPECT_EQ(grown.length(), 9);
}

TEST(IndexTest, NegativeStrideGrowKeepsOrdering) {
    ippl::Index descending(10, 2, -2);
    ippl::Index grown = descending.grow(2);

    EXPECT_EQ(grown.first(), 14);
    EXPECT_EQ(grown.last(), -2);
    EXPECT_EQ(grown.stride(), -2);
    EXPECT_EQ(grown.length(), 9);
}
