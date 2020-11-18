//
// Unit test FieldTest
//   Test the functionality of the class Field.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
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
#include "Utility/IpplException.h"

#include <cmath>
#include "gtest/gtest.h"

class FieldBCTest : public ::testing::Test {

public:
    static constexpr size_t dim = 3;
    typedef ippl::Field<double, dim> field_type;
    typedef ippl::BConds<double, dim> bc_type; 

    FieldBCTest()
    : nPoints(8)
    {
        setup();
    }

    void setup() {
        ippl::Index I(nPoints);
        NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++)
            allParallel[d] = ippl::SERIAL;

        ippl::FieldLayout<dim> layout(owned, allParallel);

        double dx = 1.0 / double(nPoints);
        ippl::Vector<double, dim> hx = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};
        ippl::UniformCartesian<double, dim> mesh(owned, hx, origin);

        field = std::make_unique<field_type>(mesh, layout);
        *field = 1.0;
        *field = (*field) * 10.0;
        HostF = field->getHostMirror();
    }

    void checkResult(const double expected) {
        Kokkos::deep_copy(HostF, field->getView());
        int N = nPoints + 2;
        for (size_t face=0; face < 2*dim; ++face) {
            size_t d = face / 2;
            switch (d) {
                case 0:
                    for(int j=0; j < N; ++j) {
                        for(int k=0; k < N; ++k) {
                            EXPECT_DOUBLE_EQ(expected, HostF(0,j,k));
                            EXPECT_DOUBLE_EQ(expected, HostF(N-1,j,k));
                        }
                     }
                    break;
                case 1:
                    for(int i=0; i < N; ++i) {
                        for(int k=0; k < N; ++k) {
                            EXPECT_DOUBLE_EQ(expected, HostF(i,0,k));
                            EXPECT_DOUBLE_EQ(expected, HostF(i,N-1,k));
                        }
                     }
                    break;
                case 2:
                    for(int i=0; i < N; ++i) {
                        for(int j=0; j < N; ++j) {
                            EXPECT_DOUBLE_EQ(expected, HostF(i,j,0));
                            EXPECT_DOUBLE_EQ(expected, HostF(i,j,N-1));
                        }
                     }
                    break;
                default:
                    throw IpplException("FieldBCTest::checkResult", 
                                         "face number wrong");
            }
        }
    }
    std::unique_ptr<field_type> field;
    bc_type bcField;
    typename field_type::view_type::host_mirror_type HostF;
    size_t nPoints;
};



TEST_F(FieldBCTest, PeriodicBC) {
    for (size_t i=0; i < 2*dim; ++i) {
        bcField[i] = std::make_shared<ippl::PeriodicFace<double, dim>>(i);
    }
    bcField.apply(*field);
    double expected = 10.0;
    checkResult(expected);
}

TEST_F(FieldBCTest, NoBC) {
    for (size_t i=0; i < 2*dim; ++i) {
        bcField[i] = std::make_shared<ippl::NoBcFace<double, dim>>(i);
    }
    bcField.apply(*field);
    double expected = 1.0;
    checkResult(expected);
}

TEST_F(FieldBCTest, ZeroBC) {
    for (size_t i=0; i < 2*dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<double, dim>>(i);
    }
    bcField.apply(*field);
    double expected = 0.0;
    checkResult(expected);
}

TEST_F(FieldBCTest, ConstantBC) {
    double constant = 7.0; 
    for (size_t i=0; i < 2*dim; ++i) {
        bcField[i] = std::make_shared<ippl::ConstantFace<double, dim>>(i, constant);
    }
    bcField.apply(*field);
    double expected = constant;
    checkResult(expected);
}

TEST_F(FieldBCTest, ExtrapolateBC) {
    for (size_t i=0; i < 2*dim; ++i) {
        bcField[i] = std::make_shared<ippl::ExtrapolateFace<double, dim>>(i, 0.0, 1.0);
    }
    bcField.apply(*field);
    double expected = 10.0;
    checkResult(expected);
}

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
