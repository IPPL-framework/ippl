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

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class FieldTest;

template <typename T, typename ExecSpace>
class FieldTest<Parameters<T, ExecSpace>> : public ::testing::Test,
                                            public MultirankUtils<1, 2, 3, 4, 5, 6> {
protected:
    void SetUp() override { CHECK_SKIP_SERIAL; }

public:
    using value_type = T;
    using exec_space = ExecSpace;

    template <unsigned Dim>
    using mesh_type = ippl::UniformCartesian<T, Dim>;

    template <unsigned Dim>
    using centering_type = typename mesh_type<Dim>::DefaultCentering;

    template <unsigned Dim>
    using field_type = ippl::Field<T, Dim, mesh_type<Dim>, centering_type<Dim>, ExecSpace>;

    template <unsigned Dim>
    using vfield_type =
        ippl::Field<ippl::Vector<T, Dim>, Dim, mesh_type<Dim>, centering_type<Dim>, ExecSpace>;

    template <unsigned Dim>
    using layout_type = ippl::FieldLayout<Dim>;

    FieldTest() {
        computeGridSizes(nPoints);
        for (unsigned d = 0; d < MaxDim; d++) {
            domain[d] = nPoints[d] / 32.;
        }
        setup(this);
    }

    template <unsigned Idx, unsigned Dim>
    void setupDim() {
        std::array<ippl::Index, Dim> indices;
        for (unsigned d = 0; d < Dim; d++) {
            indices[d] = ippl::Index(nPoints[d]);
        }
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(indices);

        ippl::Vector<T, Dim> hx;
        ippl::Vector<T, Dim> origin;

        ippl::e_dim_tag domDec[Dim];  // Specifies SERIAL, PARALLEL dims
        for (unsigned d = 0; d < Dim; d++) {
            domDec[d] = ippl::PARALLEL;
            hx[d]     = domain[d] / nPoints[d];
            origin[d] = 0;
        }

        auto layout            = std::make_shared<layout_type<Dim>>(owned, domDec);
        std::get<Idx>(layouts) = layout;

        std::get<Idx>(meshes) = std::make_shared<mesh_type<Dim>>(owned, hx, origin);

        std::get<Idx>(fields) = std::make_shared<field_type<Dim>>(*std::get<Idx>(meshes), *layout);
    }

    PtrCollection<std::shared_ptr, field_type> fields;
    PtrCollection<std::shared_ptr, mesh_type> meshes;
    PtrCollection<std::shared_ptr, layout_type> layouts;
    size_t nPoints[MaxDim];
    T domain[MaxDim];
};

template <typename Params, unsigned Dim>
struct VFieldVal {
    using vfield_view_type = typename FieldTest<Params>::template vfield_type<Dim>::view_type;
    using T                = typename FieldTest<Params>::value_type;

    const vfield_view_type vview;
    const ippl::NDIndex<Dim> lDom;

    ippl::Vector<T, Dim> dx;
    int shift;

    VFieldVal(const vfield_view_type& view, const ippl::NDIndex<Dim>& lDom, ippl::Vector<T, Dim> hx,
              int shift = 0)
        : vview(view)
        , lDom(lDom)
        , dx(hx)
        , shift(shift) {}

    template <typename... Idx>
    KOKKOS_INLINE_FUNCTION void operator()(const Idx... args) const {
        ippl::Vector<T, Dim> coords = {(T)args...};
        vview(args...)              = (0.5 + coords + lDom.first()) * dx;
    }
};

template <typename Params, unsigned Dim>
struct FieldVal {
    using field_view_type = typename FieldTest<Params>::template field_type<Dim>::view_type;
    using T               = typename FieldTest<Params>::value_type;

    const field_view_type view;

    const ippl::NDIndex<Dim> lDom;

    ippl::Vector<T, Dim> hx = 0;
    int shift;

    FieldVal(const field_view_type& view, const ippl::NDIndex<Dim>& lDom, ippl::Vector<T, Dim> hx,
             int shift = 0)
        : view(view)
        , lDom(lDom)
        , hx(hx)
        , shift(shift) {}

    // range policy tags
    struct Norm {};
    struct Integral {};
    struct Hessian {};

    const T pi = Kokkos::numbers::pi_v<T>;

    template <typename... Idx>
    KOKKOS_INLINE_FUNCTION void operator()(const Norm&, const Idx... args) const {
        T tot = (args + ...);
        for (unsigned d = 0; d < Dim; d++) {
            tot += lDom[d].first();
        }
        view(args...) = tot - 1;
    }

    template <typename... Idx>
    KOKKOS_INLINE_FUNCTION void operator()(const Integral&, const Idx... args) const {
        ippl::Vector<T, Dim> coords = {(T)args...};
        coords                      = (0.5 + coords + lDom.first() - shift) * hx;
        view(args...)               = 1;
        for (const auto& x : coords) {
            view(args...) *= Kokkos::sin(200 * pi * x);
        }
    }

    template <typename... Idx>
    KOKKOS_INLINE_FUNCTION void operator()(const Hessian&, const Idx... args) const {
        ippl::Vector<T, Dim> coords = {(T)args...};
        coords                      = (0.5 + coords + lDom.first() - shift) * hx;
        view(args...)               = 1;
        for (const auto& x : coords) {
            view(args...) *= x;
        }
    }
};

TYPED_TEST_CASE(FieldTest, MixedPrecisionAndSpaces::tests);

TYPED_TEST(FieldTest, DeepCopy) {
    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            using mirror_type =
                typename TestFixture::template field_type<Dim>::view_type::host_mirror_type;
            using field_type = typename TestFixture::template field_type<Dim>;

            *field          = 0;
            field_type copy = field->deepCopy();
            copy            = copy + 1.;

            mirror_type mirrorA = field->getHostMirror();
            mirror_type mirrorB = copy.getHostMirror();

            Kokkos::deep_copy(mirrorA, field->getView());
            Kokkos::deep_copy(mirrorB, copy.getView());

            this->template nestedViewLoop(mirrorA, field->getNghost(),
                                          [&]<typename... Idx>(const Idx... args) {
                                              assertEqual<typename TestFixture::value_type>(
                                                  mirrorA(args...) + 1, mirrorB(args...));
                                          });
        };

    this->apply(check, this->fields);
}

TYPED_TEST(FieldTest, Sum) {
    using T = typename TestFixture::value_type;

    T val                           = 1.0;
    T expected[TestFixture::MaxDim] = {val * this->nPoints[0]};
    for (unsigned d = 1; d < TestFixture::MaxDim; d++) {
        expected[d] = expected[d - 1] * this->nPoints[d];
    }

    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            *field = val;

            T sum = field->sum();

            assertEqual<T>(expected[TestFixture::dimToIndex(Dim)], sum);
        };

    this->apply(check, this->fields);
}

TYPED_TEST(FieldTest, Norm1) {
    using T = typename TestFixture::value_type;

    T val                           = -1.5;
    T expected[TestFixture::MaxDim] = {-val * this->nPoints[0]};
    for (unsigned d = 1; d < TestFixture::MaxDim; d++) {
        expected[d] = expected[d - 1] * this->nPoints[d];
    }

    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            *field = val;

            T norm1 = ippl::norm(*field, 1);

            assertEqual<T>(expected[TestFixture::dimToIndex(Dim)], norm1);
        };

    this->apply(check, this->fields);
}

TYPED_TEST(FieldTest, Norm2) {
    using T = typename TestFixture::value_type;

    T val                          = 1.5;
    T squared[TestFixture::MaxDim] = {val * val * this->nPoints[0]};
    for (unsigned d = 1; d < TestFixture::MaxDim; d++) {
        squared[d] = squared[d - 1] * this->nPoints[d];
    }

    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            *field = val;

            T norm2 = ippl::norm(*field);

            assertEqual<T>(std::sqrt(squared[TestFixture::dimToIndex(Dim)]), norm2);
        };

    this->apply(check, this->fields);
}

TYPED_TEST(FieldTest, NormInf) {
    using T = typename TestFixture::value_type;

    T val                           = 1.;
    T expected[TestFixture::MaxDim] = {this->nPoints[0] - val};
    for (unsigned d = 1; d < TestFixture::MaxDim; d++) {
        expected[d] = expected[d - 1] + this->nPoints[d];
    }

    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            using view_type = typename TestFixture::template field_type<Dim>::view_type;

            const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();

            view_type view                = field->getView();
            const ippl::Vector<T, Dim> dx = field->get_mesh().getMeshSpacing();
            FieldVal<TypeParam, Dim> fv(view, lDom, dx);
            Kokkos::parallel_for(
                "Set field",
                field->template getFieldRangePolicy<typename FieldVal<TypeParam, Dim>::Norm>(), fv);

            T normInf = ippl::norm(*field, 0);

            assertEqual<T>(expected[TestFixture::dimToIndex(Dim)], normInf);
        };

    this->apply(check, this->fields);
}

TYPED_TEST(FieldTest, VolumeIntegral) {
    using T = typename TestFixture::value_type;

    auto check = [&]<unsigned Dim>(
                     std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
        using view_type = typename TestFixture::template field_type<Dim>::view_type;

        T tol                         = (std::is_same_v<T, double>) ? 5e-15 : 5e-6;
        const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();
        const int shift               = field->getNghost();

        const ippl::Vector<T, Dim> dx = field->get_mesh().getMeshSpacing();
        view_type view                = field->getView();

        FieldVal<TypeParam, Dim> fv(view, lDom, dx, shift);
        Kokkos::parallel_for(
            "Set field",
            field->template getFieldRangePolicy<typename FieldVal<TypeParam, Dim>::Integral>(), fv);

        ASSERT_NEAR(field->getVolumeIntegral(), 0., tol);
    };

    this->apply(check, this->fields);
}

TYPED_TEST(FieldTest, VolumeIntegral2) {
    using T = typename TestFixture::value_type;

    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            *field     = 1.;
            T integral = field->getVolumeIntegral();
            T volume   = field->get_mesh().getMeshVolume();

            assertEqual<T>(integral, volume);
        };

    this->apply(check, this->fields);
}

TYPED_TEST(FieldTest, Grad) {
    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            using vfield_type = typename TestFixture::template vfield_type<Dim>;
            using view_type   = typename vfield_type::view_type;
            using mirror_type = typename view_type::host_mirror_type;

            *field = 1.;

            vfield_type vfield(field->get_mesh(), field->getLayout());
            vfield = grad(*field);

            const int shift    = vfield.getNghost();
            view_type view     = vfield.getView();
            mirror_type mirror = Kokkos::create_mirror_view(view);
            Kokkos::deep_copy(mirror, view);

            this->template nestedViewLoop(mirror, shift, [&]<typename... Idx>(const Idx... args) {
                for (size_t d = 0; d < Dim; d++) {
                    assertEqual<typename TestFixture::value_type>(mirror(args...)[d], 0.);
                }
            });
        };

    this->apply(check, this->fields);
}

TYPED_TEST(FieldTest, Div) {
    using T = typename TestFixture::value_type;

    auto check =
        [&]<unsigned Dim>(std::shared_ptr<typename TestFixture::template field_type<Dim>>& field) {
            using vfield_type = typename TestFixture::template vfield_type<Dim>;
            using vview_type  = typename vfield_type::view_type;
            using mirror_type =
                typename TestFixture::template field_type<Dim>::view_type::host_mirror_type;

            vfield_type vfield(field->get_mesh(), field->getLayout());
            vview_type view  = vfield.getView();
            const int vshift = vfield.getNghost();

            const ippl::NDIndex<Dim> lDom = vfield.getLayout().getLocalNDIndex();

            const ippl::Vector<T, Dim> dx = vfield.get_mesh().getMeshSpacing();
            VFieldVal<TypeParam, Dim> fv(view, lDom, dx, vshift);
            Kokkos::parallel_for("Set field", vfield.getFieldRangePolicy(vshift), fv);

            *field = div(vfield);

            const int shift    = field->getNghost();
            mirror_type mirror = Kokkos::create_mirror_view(field->getView());
            Kokkos::deep_copy(mirror, field->getView());

            this->template nestedViewLoop(mirror, shift, [&]<typename... Idx>(const Idx... args) {
                assertEqual<T>(mirror(args...), Dim);
            });
        };

    this->apply(check, this->fields);
}

TYPED_TEST(FieldTest, Curl) {
    using T = typename TestFixture::value_type;

    // Restrict to 3D case for now
    constexpr unsigned dim = 3;
    using mesh_type        = typename TestFixture::template mesh_type<dim>;
    using layout_type      = typename TestFixture::template layout_type<dim>;
    using vfield_type      = typename TestFixture::template vfield_type<dim>;
    using vview_type       = typename vfield_type::view_type;
    using mirror_type      = typename vview_type::host_mirror_type;

    constexpr unsigned Idx               = TestFixture::dimToIndex(dim);
    std::shared_ptr<mesh_type>& mesh     = std::get<Idx>(this->meshes);
    std::shared_ptr<layout_type>& layout = std::get<Idx>(this->layouts);

    vfield_type vfield(*mesh, *layout);
    const int nghost      = vfield.getNghost();
    vview_type view_field = vfield.getView();

    ippl::NDIndex<dim> lDom     = layout->getLocalNDIndex();
    ippl::Vector<T, dim> hx     = mesh->getMeshSpacing();
    ippl::Vector<T, dim> origin = mesh->getOrigin();

    mirror_type mirror = Kokkos::create_mirror_view(view_field);
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

                    T x = (ig + 0.5) * hx[0] + origin[0];
                    T y = (jg + 0.5) * hx[1] + origin[1];
                    T z = (kg + 0.5) * hx[2] + origin[2];

                    mirror(i, j, k)[gd] = dim0 * (y * z) + dim1 * (x * z) + dim2 * (x * y);
                }
            }
        }
    }

    Kokkos::deep_copy(view_field, mirror);

    vfield_type result(*mesh, *layout);
    result = curl(vfield);

    const int shift = result.getNghost();
    vview_type view = result.getView();
    mirror          = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
        for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
            for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                for (size_t d = 0; d < dim; ++d) {
                    assertEqual<T>(mirror(i, j, k)[d], 0.);
                }
            }
        }
    }
}

TYPED_TEST(FieldTest, Hessian) {
    using T = typename TestFixture::value_type;

    auto check = [&]<unsigned Dim>(
                     std::shared_ptr<typename TestFixture::template mesh_type<Dim>>& mesh,
                     std::shared_ptr<typename TestFixture::template layout_type<Dim>>& layout) {
        using mesh_type      = typename TestFixture::template mesh_type<Dim>;
        using centering_type = typename TestFixture::template centering_type<Dim>;
        using field_type     = typename TestFixture::template field_type<Dim>;
        using view_type      = typename field_type::view_type;
        typedef ippl::Vector<T, Dim> Vector_t;
        typedef ippl::Field<ippl::Vector<Vector_t, Dim>, Dim, mesh_type, centering_type,
                            typename TestFixture::exec_space>
            MField_t;
        using view_type_m   = typename MField_t::view_type;
        using mirror_type_m = typename view_type_m::host_mirror_type;

        field_type field(*mesh, *layout);
        int nghost           = field.getNghost();
        view_type view_field = field.getView();

        ippl::NDIndex<Dim> lDom = layout->getLocalNDIndex();
        Vector_t hx             = mesh->getMeshSpacing();
        Vector_t origin         = mesh->getOrigin();

        FieldVal<TypeParam, Dim> fv(view_field, lDom, hx, nghost);
        Kokkos::parallel_for(
            "Set field",
            field.template getFieldRangePolicy<typename FieldVal<TypeParam, Dim>::Hessian>(nghost),
            fv);

        MField_t result(*mesh, *layout);
        result = hess(field);

        nghost                      = result.getNghost();
        view_type_m view_result     = result.getView();
        mirror_type_m mirror_result = Kokkos::create_mirror_view(view_result);
        Kokkos::deep_copy(mirror_result, view_result);

        this->template nestedViewLoop(mirror_result, nghost,
                                      [&]<typename... Idx>(const Idx... args) {
                                          T det = 0;
                                          for (unsigned d = 0; d < Dim; d++) {
                                              det += mirror_result(args...)[d][d];
                                          }
                                          assertEqual<T>(det, 0.);
                                      });
    };

    this->apply(check, this->meshes, this->layouts);
}

int main(int argc, char* argv[]) {
    int success = 1;
    MixedPrecisionAndSpaces::checkArgs(argc, argv);
    ippl::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
