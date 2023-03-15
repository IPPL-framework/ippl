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
    typedef ippl::UniformCartesian<double, dim> mesh_type;
    typedef ippl::Field<float, dim> field_type;
    typedef ippl::FieldLayout<dim> layout_type;

    FieldTest()
    : nPoints(8)
    {
        setup();
    }

    void setup() {
        ippl::Index I(nPoints);
        ippl::NDIndex<dim> owned(I, I, I);

        ippl::e_dim_tag domDec[dim];    // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++)
            domDec[d] = ippl::PARALLEL;

        layout = std::make_shared<layout_type>(owned, domDec);

        double dx = 1.0 / double(nPoints);
        ippl::Vector<double, dim> hx = {dx, dx, dx};
        ippl::Vector<double, dim> origin = {0, 0, 0};
        mesh = std::make_shared<mesh_type>(owned, hx, origin);

        field = std::make_unique<field_type>(*mesh, *layout);
    }

    std::unique_ptr<field_type> field;
    std::shared_ptr<mesh_type> mesh;
    std::shared_ptr<layout_type> layout;
    size_t nPoints;
};


TEST_F(FieldTest, Norm1) {
    float val = -1.5;

    *field = val;

    float norm1 = ippl::norm(*field, 1);

    ASSERT_FLOAT_EQ(-val * std::pow(nPoints, dim), norm1);
}


TEST_F(FieldTest, Norm2) {
    float val = 1.5;

    *field = val;

    float norm2 = ippl::norm(*field);

    ASSERT_FLOAT_EQ(std::sqrt(val * val * std::pow(nPoints, dim)), norm2);
}

TEST_F(FieldTest, NormInf) {
    const ippl::NDIndex<dim> lDom = field->getLayout().getLocalNDIndex();
    const int shift = field->getNghost();

    auto view = field->getView();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                const size_t ig = i + lDom[0].first();
                const size_t jg = j + lDom[1].first();
                const size_t kg = k + lDom[2].first();

                mirror(i, j, k) = -1.0 + (ig + jg + kg);
            }
        }
    }
    Kokkos::deep_copy(view, mirror);


    float normInf = ippl::norm(*field, 0);

    float val = -1.0 + 3 * nPoints;

    ASSERT_FLOAT_EQ(val, normInf);
}

TEST_F(FieldTest, VolumeIntegral) {
    const ippl::NDIndex<dim> lDom = field->getLayout().getLocalNDIndex();
    const int shift = field->getNghost();

    const float dx = 1. / nPoints;
    auto view = field->getView();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);
    const float pi = acos(-1.0);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                const size_t ig = i + lDom[0].first() - shift;
                const size_t jg = j + lDom[1].first() - shift;
                const size_t kg = k + lDom[2].first() - shift;
                float x = (ig + 0.5) * dx;
                float y = (jg + 0.5) * dx;
                float z = (kg + 0.5) * dx;

                mirror(i, j, k) = sin(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z);
            }
        }
    }
    Kokkos::deep_copy(view, mirror);

    ASSERT_NEAR(field->getVolumeIntegral(), 0., 1e-6);
}

TEST_F(FieldTest, VolumeIntegral2) {
    *field = 1.;
    float integral = field->getVolumeIntegral();
    double volume = field->get_mesh().getMeshVolume();
    ASSERT_FLOAT_EQ(integral, volume);
}

TEST_F(FieldTest, Grad) {
    *field = 1.;

    ippl::Field<ippl::Vector<float, dim>, dim> vfield(*mesh, *layout);
    vfield = grad(*field);

    const int shift = vfield.getNghost();
    auto view = vfield.getView();
    auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                for (size_t d = 0; d < dim; ++d) {
                    ASSERT_FLOAT_EQ(mirror(i, j, k)[d], 0.);
                }
            }
        }
    }
}

TEST_F(FieldTest, Curl) {

    ippl::Field<ippl::Vector<float, dim>, dim, mesh_type> vfield(*mesh, *layout);
    const int nghost = vfield.getNghost();
    auto view_field = vfield.getView();
    
    auto lDom = this->layout->getLocalNDIndex();
    ippl::Vector<float, dim> hx = this->mesh->getMeshSpacing();
    ippl::Vector<float, dim> origin = this->mesh->getOrigin();   

    auto mirror = Kokkos::create_mirror_view(view_field);
    Kokkos::deep_copy(mirror, view_field);

    for (unsigned int gd = 0; gd < dim; ++gd) {
        
        bool dim0 = (gd == 0);
        bool dim1 = (gd == 1);
        bool dim2 = (gd == 2);

        for (size_t i = 0; i < view_field.extent(0); ++i) {
            for (size_t j = 0; j < view_field.extent(1); ++j) {
                for (size_t k = 0; k < view_field.extent(2); ++k) {

                    //local to global index conversion
                    const int ig = i + lDom[0].first() - nghost;
                    const int jg = j + lDom[1].first() - nghost;
                    const int kg = k + lDom[2].first() - nghost;
            
                    double x = (ig + 0.5) * hx[0] + origin[0];
                    double y = (jg + 0.5) * hx[1] + origin[1];
                    double z = (kg + 0.5) * hx[2] + origin[2];
                
                    mirror(i,j,k)[gd] = dim0 * (y*z) +
                                        dim1 * (x*z) +
                                        dim2 * (x*y);
                }
            }
        }
    }

    Kokkos::deep_copy(view_field, mirror);

    ippl::Field<ippl::Vector<float, dim>, dim> result(*mesh, *layout);
    result = curl(vfield);

    const int shift = result.getNghost();
    auto view = result.getView();
    mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                for (size_t d = 0; d < dim; ++d) {
                    ASSERT_FLOAT_EQ(mirror(i, j, k)[d], 0.);
                }
            }
        }
    }
}

TEST_F(FieldTest, Hessian) {

    typedef ippl::Vector<float, dim> Vector_t;
    typedef ippl::Field<ippl::Vector<Vector_t,dim>, dim> MField_t;

    ippl::Field<float, dim, mesh_type> field(*mesh, *layout);
    int nghost = field.getNghost();
    auto view_field = field.getView();
    
    auto lDom = this->layout->getLocalNDIndex();
    ippl::Vector<double, dim> hx = this->mesh->getMeshSpacing();
    ippl::Vector<double, dim> origin = this->mesh->getOrigin();   

    auto mirror = Kokkos::create_mirror_view(view_field);
    Kokkos::deep_copy(mirror, view_field);

    for (size_t i = 0; i < view_field.extent(0); ++i) {
        for (size_t j = 0; j < view_field.extent(1); ++j) {
            for (size_t k = 0; k < view_field.extent(2); ++k) {

                    //local to global index conversion
                    const int ig = i + lDom[0].first() - nghost;
                    const int jg = j + lDom[1].first() - nghost;
                    const int kg = k + lDom[2].first() - nghost;
            
                    double x = (ig + 0.5) * hx[0] + origin[0];
                    double y = (jg + 0.5) * hx[1] + origin[1];
                    double z = (kg + 0.5) * hx[2] + origin[2];
                
                    mirror(i,j,k) = x*y*z;
            }
        }
    }

    Kokkos::deep_copy(view_field, mirror);

    MField_t result(*mesh, *layout);
    result = hess(field);

    nghost = result.getNghost();
    auto view_result = result.getView();
    auto mirror_result = Kokkos::create_mirror_view(view_result);
    Kokkos::deep_copy(mirror_result, view_result);

    for (size_t i = nghost; i < view_result.extent(0)-nghost; ++i) {
        for (size_t j = nghost; j < view_result.extent(1)-nghost; ++j) {
            for (size_t k = nghost; k < view_result.extent(2)-nghost; ++k) {

                float det = mirror_result(i,j,k)[0][0] + 
                             mirror_result(i,j,k)[1][1] +
                             mirror_result(i,j,k)[2][2] ;
            
                ASSERT_FLOAT_EQ(det, 0.);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    Ippl ippl(argc,argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
