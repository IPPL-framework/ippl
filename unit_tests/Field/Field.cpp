//
// Unit test FieldTest
//   Test the functionality of the classes Field and BareField.
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
    typedef ippl::UniformCartesian<double, dim> mesh_type;

    FieldTest()
    : nPoints(8)
    {
        setup();
    }

    void setup() {
        ippl::Index I(nPoints);
        ippl::NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag allParallel[dim];    // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++)
            allParallel[d] = ippl::SERIAL;

        ippl::FieldLayout<dim> layout(owned, allParallel);

        double dx = 1.0 / double(nPoints);
        ippl::Vector<double, dim> hx = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};
        mesh = std::make_shared<mesh_type>(owned, hx, origin);

        field = std::make_unique<field_type>(*mesh, layout);
    }

    std::unique_ptr<field_type> field;
    std::shared_ptr<mesh_type> mesh;
    size_t nPoints;
};



TEST_F(FieldTest, Sum) {
    double val = 1.0;

    *field = val;

    double sum = field->sum();

    ASSERT_DOUBLE_EQ(val * std::pow(nPoints, dim), sum);
}

TEST_F(FieldTest, Min) {
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

    double min = field->min();
    // minimum value -1 + nghost + nghost + nghost
    ASSERT_DOUBLE_EQ(min, 2.);
}

TEST_F(FieldTest, Max) {
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

    double max = field->max();
    double expected = -1. + nPoints * 3;
    ASSERT_DOUBLE_EQ(max, expected);
}

TEST_F(FieldTest, Prod) {
    *field = 2.;
    double val = field->prod();
    ASSERT_DOUBLE_EQ(val, pow(2, nPoints * nPoints * nPoints));
}

TEST_F(FieldTest, ScalarMultiplication) {
    *field = 1.;
    *field = *field * 10;

    auto view = field->getView();
    const int nghost = field->getNghost();
    for (size_t i = nghost; i < nPoints - nghost; ++i) {
        for (size_t j = nghost; j < nPoints - nghost; ++j) {
            for (size_t k = nghost; k < nPoints - nghost; ++k) {
                ASSERT_DOUBLE_EQ(view(i,j,k), 10.);
            }
        }
    }
}


TEST_F(FieldTest, Norm1) {
    double val = -1.5;

    *field = val;

    double norm1 = ippl::norm(*field, 1);

    ASSERT_DOUBLE_EQ(-val * std::pow(nPoints, dim), norm1);
}


TEST_F(FieldTest, Norm2) {
    double val = 1.5;

    *field = val;

    double norm2 = ippl::norm(*field);

    ASSERT_DOUBLE_EQ(std::sqrt(val * val * std::pow(nPoints, dim)), norm2);
}

TEST_F(FieldTest, NormInf) {

    const ippl::NDIndex<dim> lDom = field->getLayout().getLocalNDIndex();

    auto view = field->getView();
    auto policy = field->getRangePolicy();

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


    double normInf = ippl::norm(*field, 0);

    double val = -1.0 + 3 * nPoints;

    ASSERT_DOUBLE_EQ(val, normInf);
}

TEST_F(FieldTest, VolumeIntegral) {
    const double dx = 1. / nPoints;
    auto view = field->getView();
    auto policy = field->getRangePolicy();
    const double pi = acos(-1.0);

    Kokkos::parallel_for("assign field", policy,
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
            double x = (i - 0.5) * dx;
            double y = (j - 0.5) * dx;
            double z = (k - 0.5) * dx;

            view(i, j, k) = sin(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z);
        }
    );

    ASSERT_NEAR(field->getVolumeIntegral(), 0., 1e-15);
}

TEST_F(FieldTest, VolumeIntegral2) {
    *field = 1.;
    double integral = field->getVolumeIntegral();
    double volume = field->get_mesh().getMeshVolume();
    ASSERT_DOUBLE_EQ(integral, volume);
}


int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
