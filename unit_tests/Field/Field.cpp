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

#include "MultirankUtils.h"
#include "gtest/gtest.h"

class FieldTest : public ::testing::Test, public MultirankUtils<1, 2, 3, 4, 5, 6> {
public:
    template <unsigned Dim>
    using field_type = ippl::Field<double, Dim>;

    template <unsigned Dim>
    using mesh_type = ippl::UniformCartesian<double, Dim>;

    template <unsigned Dim>
    using layout_type = ippl::FieldLayout<Dim>;

    FieldTest()
        : nPoints(8) {
        setup(this);
    }

    template <unsigned Idx, unsigned Dim>
    void setupDim() {
        ippl::Index I(nPoints);
        std::array<ippl::Index, Dim> args;
        args.fill(I);
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(args);

        double dx = 1.0 / double(nPoints);
        ippl::Vector<double, Dim> hx;
        ippl::Vector<double, Dim> origin;

        ippl::e_dim_tag domDec[Dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned d = 0; d < Dim; d++) {
            domDec[d] = ippl::PARALLEL;
            hx[d]     = dx;
            origin[d] = 0;
        }

        auto layout            = std::make_shared<layout_type<Dim>>(owned, domDec);
        std::get<Idx>(layouts) = layout;

        std::get<Idx>(meshes) = std::make_shared<mesh_type<Dim>>(owned, hx, origin);

        std::get<Idx>(fields) = std::make_unique<field_type<Dim>>(*std::get<Idx>(meshes), *layout);
    }

    PtrCollection<std::shared_ptr, field_type> fields;
    PtrCollection<std::shared_ptr, mesh_type> meshes;
    PtrCollection<std::shared_ptr, layout_type> layouts;
    size_t nPoints;
};

TEST_F(FieldTest, Sum) {
    double val = 1.0;

    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field = val;

        double sum = field->sum();

        ASSERT_DOUBLE_EQ(val * std::pow(nPoints, Dim), sum);
    };

    apply(check, fields);
}

TEST_F(FieldTest, Norm1) {
    double val = -1.5;

    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field = val;

        double norm1 = ippl::norm(*field, 1);

        ASSERT_DOUBLE_EQ(-val * std::pow(nPoints, Dim), norm1);
    };

    apply(check, fields);
}

TEST_F(FieldTest, Norm2) {
    double val = 1.5;

    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field = val;

        double norm2 = ippl::norm(*field);

        ASSERT_DOUBLE_EQ(std::sqrt(val * val * std::pow(nPoints, Dim)), norm2);
    };

    apply(check, fields);
}

TEST_F(FieldTest, NormInf) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();

        auto view = field->getView();
        Kokkos::parallel_for(
            "Set field", field->getRangePolicy(),
            KOKKOS_LAMBDA<typename... Idx>(const Idx... args) {
                double tot = (args + ...);
                for (unsigned d = 0; d < Dim; d++)
                    tot += lDom[d].first();
                view(args...) = tot - 1;
            });

        double normInf = ippl::norm(*field, 0);

        double val = -1.0 + Dim * nPoints;

        ASSERT_DOUBLE_EQ(val, normInf);
    };

    apply(check, fields);
}

TEST_F(FieldTest, VolumeIntegral) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();
        const int shift               = field->getNghost();

        const double dx = 1. / nPoints;
        auto view       = field->getView();
        const double pi = acos(-1.0);

        auto toCoords = KOKKOS_LAMBDA<size_t D>(size_t x)->double {
            return (x + lDom[D].first() - shift + 0.5) * dx;
        };

        auto fieldVal = KOKKOS_LAMBDA<size_t... Dims, typename... Idx>(std::index_sequence<Dims...>,
                                                                       const Idx... args) {
            return ((sin(2 * pi * toCoords.template operator()<Dims>(args))) * ...);
        };

        Kokkos::parallel_for(
            "Set field", field->getRangePolicy(),
            KOKKOS_LAMBDA<typename... Idx>(const Idx... args) {
                view(args...) = fieldVal(std::make_index_sequence<Dim>{}, args...);
            });

        ASSERT_NEAR(field->getVolumeIntegral(), 0., 1e-15);
    };

    apply(check, fields);
}

TEST_F(FieldTest, VolumeIntegral2) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field          = 1.;
        double integral = field->getVolumeIntegral();
        double volume   = field->get_mesh().getMeshVolume();
        ASSERT_DOUBLE_EQ(integral, volume);
    };

    apply(check, fields);
}

TEST_F(FieldTest, Grad) {
    auto check = [&]<unsigned Dim>(std::shared_ptr<field_type<Dim>>& field) {
        *field = 1.;

        ippl::Field<ippl::Vector<double, Dim>, Dim> vfield(field->get_mesh(), field->getLayout());
        vfield = grad(*field);

        const int shift = vfield.getNghost();
        auto view       = vfield.getView();
        auto mirror     = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(mirror, view);

        nestedViewLoop<Dim>(mirror, shift, [&]<typename... Idx>(const Idx... args) {
            for (size_t d = 0; d < Dim; d++)
                ASSERT_DOUBLE_EQ(mirror(args...)[d], 0.);
        });
    };

    apply(check, fields);
}

TEST_F(FieldTest, Curl) {
    // Restrict to 3D case for now
    auto mesh              = std::get<2>(meshes);
    auto layout            = std::get<2>(layouts);
    constexpr unsigned dim = 3;

    ippl::Field<ippl::Vector<double, dim>, dim> vfield(*mesh, *layout);
    const int nghost = vfield.getNghost();
    auto view_field  = vfield.getView();

    auto lDom                        = layout->getLocalNDIndex();
    ippl::Vector<double, dim> hx     = mesh->getMeshSpacing();
    ippl::Vector<double, dim> origin = mesh->getOrigin();

    auto mirror = Kokkos::create_mirror_view(view_field);
    Kokkos::deep_copy(mirror, view_field);

    for (unsigned int gd = 0; gd < dim; ++gd) {
        bool dim0 = (gd == 0);
        bool dim1 = (gd == 1);
        bool dim2 = (gd == 2);

        for (size_t i = 0; i < view_field.extent(0); ++i) {
            for (size_t j = 0; j < view_field.extent(1); ++j) {
                for (size_t k = 0; k < view_field.extent(2); ++k) {
                    // local to global index conversion
                    const int ig = i + lDom[0].first() - nghost;
                    const int jg = j + lDom[1].first() - nghost;
                    const int kg = k + lDom[2].first() - nghost;

                    double x = (ig + 0.5) * hx[0] + origin[0];
                    double y = (jg + 0.5) * hx[1] + origin[1];
                    double z = (kg + 0.5) * hx[2] + origin[2];

                    mirror(i, j, k)[gd] = dim0 * (y * z) + dim1 * (x * z) + dim2 * (x * y);
                }
            }
        }
    }

    Kokkos::deep_copy(view_field, mirror);

    ippl::Field<ippl::Vector<double, dim>, dim> result(*mesh, *layout);
    result = curl(vfield);

    const int shift = result.getNghost();
    auto view       = result.getView();
    mirror          = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                for (size_t d = 0; d < dim; ++d) {
                    ASSERT_DOUBLE_EQ(mirror(i, j, k)[d], 0.);
                }
            }
        }
    }
}

TEST_F(FieldTest, Hessian) {
    // Restrict to 3D case for now
    auto mesh              = std::get<2>(meshes);
    auto layout            = std::get<2>(layouts);
    constexpr unsigned dim = 3;

    typedef ippl::Vector<double, dim> Vector_t;
    typedef ippl::Field<ippl::Vector<Vector_t, dim>, dim> MField_t;

    ippl::Field<double, dim> field(*mesh, *layout);
    int nghost      = field.getNghost();
    auto view_field = field.getView();

    auto lDom                        = layout->getLocalNDIndex();
    ippl::Vector<double, dim> hx     = mesh->getMeshSpacing();
    ippl::Vector<double, dim> origin = mesh->getOrigin();

    auto mirror = Kokkos::create_mirror_view(view_field);
    Kokkos::deep_copy(mirror, view_field);

    for (size_t i = 0; i < view_field.extent(0); ++i) {
        for (size_t j = 0; j < view_field.extent(1); ++j) {
            for (size_t k = 0; k < view_field.extent(2); ++k) {
                // local to global index conversion
                const int ig = i + lDom[0].first() - nghost;
                const int jg = j + lDom[1].first() - nghost;
                const int kg = k + lDom[2].first() - nghost;

                double x = (ig + 0.5) * hx[0] + origin[0];
                double y = (jg + 0.5) * hx[1] + origin[1];
                double z = (kg + 0.5) * hx[2] + origin[2];

                mirror(i, j, k) = x * y * z;
            }
        }
    }

    Kokkos::deep_copy(view_field, mirror);

    MField_t result(*mesh, *layout);
    result = hess(field);

    nghost             = result.getNghost();
    auto view_result   = result.getView();
    auto mirror_result = Kokkos::create_mirror_view(view_result);
    Kokkos::deep_copy(mirror_result, view_result);

    for (size_t i = nghost; i < view_result.extent(0) - nghost; ++i) {
        for (size_t j = nghost; j < view_result.extent(1) - nghost; ++j) {
            for (size_t k = nghost; k < view_result.extent(2) - nghost; ++k) {
                double det = mirror_result(i, j, k)[0][0] + mirror_result(i, j, k)[1][1]
                             + mirror_result(i, j, k)[2][2];

                ASSERT_DOUBLE_EQ(det, 0.);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
