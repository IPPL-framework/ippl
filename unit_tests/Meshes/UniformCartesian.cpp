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

#include "MultirankUtils.h"
#include "gtest/gtest.h"

class UniformCartesianTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    UniformCartesianTest() {}
};

TEST_F(UniformCartesianTest, Constructor) {
    auto check = [&]<unsigned Dim>() {
        int pt = 10;
        ippl::Index I(pt);
        std::array<ippl::Index, Dim> args;
        args.fill(I);
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        double dx = 1.0 / double(pt);
        ippl::Vector<double, Dim> hx;
        ippl::Vector<double, Dim> origin;
        for (unsigned d = 0; d < Dim; d++) {
            hx[d]     = dx;
            origin[d] = 0;
        }
        ippl::UniformCartesian<double, Dim> mesh(owned, hx, origin);

        double length = mesh.getCellVolume();

        ASSERT_DOUBLE_EQ(length, pow(dx, Dim));
        ASSERT_DOUBLE_EQ(mesh.getMeshVolume(), 1.);
    };

    apply(check);
}

TEST_F(UniformCartesianTest, Initialize) {
    auto check = [&]<unsigned Dim>() {
        int pt = 10;
        ippl::Index I(pt);
        std::array<ippl::Index, Dim> args;
        args.fill(I);
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        double dx = 1.0 / double(pt);
        ippl::Vector<double, Dim> hx;
        ippl::Vector<double, Dim> origin;
        for (unsigned d = 0; d < Dim; d++) {
            hx[d]     = dx;
            origin[d] = 0;
        }

        ippl::UniformCartesian<double, Dim> mesh;
        mesh.initialize(owned, hx, origin);

        double volume = mesh.getCellVolume();

        ASSERT_DOUBLE_EQ(volume, pow(dx, Dim));
        ASSERT_DOUBLE_EQ(mesh.getMeshVolume(), 1.);
    };

    apply(check);
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
