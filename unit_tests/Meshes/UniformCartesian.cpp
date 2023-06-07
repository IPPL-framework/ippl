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

#include "MultirankUtils.h"
#include "gtest/gtest.h"

template <typename T>
class UniformCartesianTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    UniformCartesianTest() { computeGridSizes(nPoints); }

    template <unsigned Dim>
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

    size_t nPoints[MaxDim];
};

using Precisions = ::testing::Types<double, float>;

TYPED_TEST_CASE(UniformCartesianTest, Precisions);

TYPED_TEST(UniformCartesianTest, Constructor) {
    auto check = [&]<unsigned Dim>() {
        ippl::Vector<TypeParam, Dim> hx;
        ippl::Vector<TypeParam, Dim> origin;
        TypeParam cellVol, meshVol;

        ippl::NDIndex<Dim> owned = this->createMesh(hx, origin, cellVol, meshVol);
        ippl::UniformCartesian<TypeParam, Dim> mesh(owned, hx, origin);

        TypeParam length = mesh.getCellVolume();

        if constexpr (std::is_same_v<TypeParam, double>) {
            ASSERT_DOUBLE_EQ(length, cellVol);
            ASSERT_DOUBLE_EQ(mesh.getMeshVolume(), meshVol);
        } else {
            ASSERT_FLOAT_EQ(length, cellVol);
            ASSERT_FLOAT_EQ(mesh.getMeshVolume(), meshVol);
        }
    };

    this->apply(check);
}

TYPED_TEST(UniformCartesianTest, Initialize) {
    auto check = [&]<unsigned Dim>() {
        ippl::Vector<TypeParam, Dim> hx;
        ippl::Vector<TypeParam, Dim> origin;
        TypeParam cellVol, meshVol;

        ippl::NDIndex<Dim> owned = this->createMesh(hx, origin, cellVol, meshVol);

        ippl::UniformCartesian<TypeParam, Dim> mesh;
        mesh.initialize(owned, hx, origin);

        if constexpr (std::is_same_v<TypeParam, double>) {
            ASSERT_DOUBLE_EQ(mesh.getCellVolume(), cellVol);
            ASSERT_DOUBLE_EQ(mesh.getMeshVolume(), meshVol);
        } else {
            ASSERT_FLOAT_EQ(mesh.getCellVolume(), cellVol);
            ASSERT_FLOAT_EQ(mesh.getMeshVolume(), meshVol);
        }
    };

    this->apply(check);
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
