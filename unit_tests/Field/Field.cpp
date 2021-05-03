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

#include <cmath>
#include "gtest/gtest.h"

class FieldTest : public ::testing::Test {

public:
    static constexpr size_t dim = 3;
    typedef ippl::Field<double, dim> field_type;

    FieldTest()
    : nPoints(8)
    {
        setup();
    }

    void setup() {
        ippl::Index I(nPoints);
        ippl::NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag allParallel[dim];
        for (unsigned int d = 0; d < dim; d++)
            allParallel[d] = ippl::PARALLEL;

        ippl::FieldLayout<dim> layout(owned, allParallel);

        double dx = 1.0 / double(nPoints);
        ippl::Vector<double, dim> hx = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};
        ippl::UniformCartesian<double, dim> mesh(owned, hx, origin);

        field = std::make_unique<field_type>(mesh, layout);
    }

    std::unique_ptr<field_type> field;
    size_t nPoints;
};



TEST_F(FieldTest, Sum) {
    double val = 1.0;

    *field = val;

    double sum = field->sum();

    ASSERT_DOUBLE_EQ(val * std::pow(nPoints, dim), sum);
}


TEST_F(FieldTest, Norm1) {
    double val = -1.5;

    *field = val;

    double norm1 = ippl::norm1(*field);

    ASSERT_DOUBLE_EQ(-val * std::pow(nPoints, dim), norm1);
}


TEST_F(FieldTest, Norm2) {
    double val = 1.5;

    *field = val;

    double norm2 = ippl::norm2(*field);

    ASSERT_DOUBLE_EQ(std::sqrt(val * val * std::pow(nPoints, dim)), norm2);
}

TEST_F(FieldTest, NormInf) {

    const ippl::NDIndex<dim> lDom = field->getLayout().getLocalNDIndex();

    auto view = field->getView();
    auto policy = field->getRangePolicy(0);

    Kokkos::parallel_for("Assign field",
                         policy,
                         KOKKOS_LAMBDA(const int i,
                                       const int j,
                                       const int k)
    {
        const size_t ig = i + lDom[0].first();
        const size_t jg = j + lDom[1].first();
        const size_t kg = k + lDom[2].first();

        view(i, j, k) = -1.0 + (ig + jg + kg);
    });


    double normInf = ippl::normInf(*field);

    double val = -1.0 + 3 * nPoints;

    ASSERT_DOUBLE_EQ(val, normInf);
}




int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
