//
// Unit test UniformCartesianTest
//   Test functionality of the class UniformCartesian.
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

#include <cmath>
#include "gtest/gtest.h"

class UniformCartesianTest : public ::testing::Test {

public:
    UniformCartesianTest()
    { }
};


TEST_F(UniformCartesianTest, Constructor3DDp) {
    int pt = 10;
    ippl::Index I(pt);
    ippl::NDIndex<3> owned(I, I, I);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);

    double volume = mesh.getCellVolume();

    ASSERT_DOUBLE_EQ(volume, dx * dx * dx);
    ASSERT_DOUBLE_EQ(mesh.getMeshVolume(), 1.);
}


TEST_F(UniformCartesianTest, Constructor3DSp) {
    int pt = 10;
    ippl::Index I(pt);
    ippl::NDIndex<3> owned(I, I, I);

    float  dx = 1.0 / float(pt);
    ippl::Vector<float, 3> hx = {dx, dx, dx};
    ippl::Vector<float, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<float, 3> mesh(owned, hx, origin);

    float volume = mesh.getCellVolume();
    
    ASSERT_FLOAT_EQ(volume, dx * dx * dx);
    ASSERT_FLOAT_EQ(mesh.getMeshVolume(), 1.);

}


TEST_F(UniformCartesianTest, Initialize3DDp) {
    int pt = 10;
    ippl::Index I(pt);
    ippl::NDIndex<3> owned(I, I, I);

    double dx = 1.0 / double(pt);
    ippl::Vector<double, 3> hx = {dx, dx, dx};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<double, 3> mesh;

    mesh.initialize(owned, hx, origin);

    double area = mesh.getCellVolume();

    ASSERT_DOUBLE_EQ(area, dx * dx * dx);
    ASSERT_DOUBLE_EQ(mesh.getMeshVolume(), 1.);
}


TEST_F(UniformCartesianTest, Initialize3DSp) {
    int pt = 10;
    ippl::Index I(pt);
    ippl::NDIndex<3> owned(I, I, I);

    float dx = 1.0 / float(pt);
    ippl::Vector<float, 3> hx = {dx, dx, dx};
    ippl::Vector<float, 3> origin = {0, 0, 0};
    ippl::UniformCartesian<float, 3> mesh;

    mesh.initialize(owned, hx, origin);

    float volume = mesh.getCellVolume();

    ASSERT_FLOAT_EQ(volume, dx * dx * dx);
    ASSERT_FLOAT_EQ(mesh.getMeshVolume(), 1.);
}


int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
