//
// Unit test ParameterListTest
//   Test functionality of the class ParameterList.
//
// Copyright (c) 2021, Matthias Frey, University of St Andrews, St Andrews, Scotland
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

#include "Utility/ParameterList.h"
#include "Utility/IpplException.h"
#include "gtest/gtest.h"

class ParameterListTest : public ::testing::Test {

public:

    ParameterListTest() { }
};


TEST_F(ParameterListTest, Add) {

    ippl::ParameterList p;
    p.add<double>("tolerance", 1.0e-8);

    bool isContained = false;
    try {
        p.add<double>("tolerance", 1.0e-9);
    } catch(...) {
        isContained = true;
    }

    EXPECT_EQ(isContained, true);
}


TEST_F(ParameterListTest, UpdateSingle) {

    double tol = 1.0e-6;

    ippl::ParameterList p;
    p.add<double>("tolerance", 1.0e-9);

    p.update("tolerance", tol);

    EXPECT_EQ(p.get<double>("tolerance"), tol);


    bool isContained = true;
    try {
        p.update<bool>("enable", true);
    } catch(...) {
        isContained = false;
    }

    EXPECT_EQ(isContained, false);
}


TEST_F(ParameterListTest, Merge) {

    ippl::ParameterList p1;
    p1.add<double>("tolerance", 1.0e-8);
    p1.add<bool>("is enabled", false);

    ippl::ParameterList p2;

    double tol = 1.0e-12;
    int size = 5;

    p2.add<int>("size", size);
    p2.add<double>("tolerance", tol);

    p1.merge(p2);

    EXPECT_EQ(p1.get<double>("tolerance"), tol);
    EXPECT_EQ(p1.get<int>("size"), size);
    EXPECT_EQ(p1.get<bool>("is enabled"), false);
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

    bool isContained = false;
    try {
        int size = p1.get<int>("size");
        size += 1;
        isContained = true;
    } catch(...) {
        // do nothing here
    }

    EXPECT_EQ(p1.get<double>("tolerance"), tol);
    EXPECT_EQ(isContained, false);
    EXPECT_EQ(p1.get<bool>("is enabled"), false);
}


int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
