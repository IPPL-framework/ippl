//
// Unit test FieldSubLayoutTest
//   Test the functionality of the class Field using a SubFieldLayout.
//
#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "../src/PoissonSolvers/LaplaceHelpers.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

template <typename>
class FieldSubLayoutTest;

template <typename T, typename ExecSpace, unsigned Dim>
class FieldSubLayoutTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type              = T;
    using exec_space              = ExecSpace;
    constexpr static unsigned dim = Dim;

    using mesh_type      = ippl::UniformCartesian<T, Dim>;
    using centering_type = typename mesh_type::DefaultCentering;
    using field_type     = ippl::Field<T, Dim, mesh_type, centering_type, ExecSpace>;
    using vfield_type =
    ippl::Field<ippl::Vector<T, Dim>, Dim, mesh_type, centering_type, ExecSpace>;
    using layout_type = ippl::SubFieldLayout<Dim>;
    using testlayout_type = ippl::FieldLayout<Dim>;

    FieldSubLayoutTest()
        : nPoints(getGridSizes<Dim>()) {

        // Calculate the domain size of the original layout
        // In this test we use a domain of +1 Point in each direction,
        // which is the domain of the origin layout,
        // used to determine the distribution of the sub-domains on the ranks

        unsigned int subDomainReduction = 1;

        for (unsigned d = 0; d < Dim; d++) {
            domain[d] = (nPoints[d]+subDomainReduction) / 32.;
        }

        std::array<ippl::Index, Dim> originIndices;
        std::array<ippl::Index, Dim> indices;
        for (unsigned d = 0; d < Dim; d++) {
            originIndices[d] = ippl::Index(nPoints[d]+subDomainReduction);
            indices[d] = ippl::Index(nPoints[d]);
        }
        auto originOwned = std::make_from_tuple<ippl::NDIndex<Dim>>(originIndices);
        auto owned = std::make_from_tuple<ippl::NDIndex<Dim>>(indices);

        ippl::Vector<T, Dim> hx;
        ippl::Vector<T, Dim> origin;

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        for (unsigned d = 0; d < Dim; d++) {
            hx[d]     = domain[d] / (nPoints[d]+subDomainReduction);
            origin[d] = 0;
        }

        layout = std::make_shared<layout_type>(MPI_COMM_WORLD, originOwned, owned, isParallel);
        mesh   = std::make_shared<mesh_type>(owned, hx, origin);
        field  = std::make_shared<field_type>(*mesh, *layout);
        
    }

    std::shared_ptr<field_type> field;
    std::shared_ptr<mesh_type> mesh;
    std::shared_ptr<layout_type> layout;
    std::array<size_t, Dim> nPoints;
    std::array<T, Dim> domain;
};

template <typename Params>
struct VFieldVal {
    using vfield_view_type        = typename FieldSubLayoutTest<Params>::vfield_type::view_type;
    using T                       = typename FieldSubLayoutTest<Params>::value_type;
    constexpr static unsigned Dim = FieldSubLayoutTest<Params>::dim;

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
        ippl::Vector<T, Dim> coords = {static_cast<T>(args)...};
        vview(args...)              = (0.5 + coords + lDom.first()) * dx;
    }
};

template <typename Params>
struct FieldVal {
    using field_view_type         = typename FieldSubLayoutTest<Params>::field_type::view_type;
    using T                       = typename FieldSubLayoutTest<Params>::value_type;
    constexpr static unsigned Dim = FieldSubLayoutTest<Params>::dim;

    const field_view_type view;

    const ippl::NDIndex<Dim> lDom;

    ippl::Vector<T, Dim> hx   = 0;
    ippl::Vector<T, Dim> rmax = 0;
    int shift;

    FieldVal(const field_view_type& view, const ippl::NDIndex<Dim>& lDom, ippl::Vector<T, Dim> hx,
             int shift = 0, ippl::Vector<T, Dim> rmax = 0)
        : view(view)
        , lDom(lDom)
        , hx(hx)
        , rmax(rmax)
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
        ippl::Vector<T, Dim> coords = {static_cast<T>(args)...};
        coords                      = (0.5 + coords + lDom.first() - shift) * hx;
        view(args...)               = 1;
        for (const auto& x : coords) {
            view(args...) *= Kokkos::sin(200 * pi * x);
        }
    }

    template <typename... Idx>
    KOKKOS_INLINE_FUNCTION void operator()(const Hessian&, const Idx... args) const {
        ippl::Vector<T, Dim> coords = {static_cast<T>(args)...};
        coords                      = (0.5 + coords + lDom.first() - shift) * hx;
        view(args...)               = 1;
        for (const auto& x : coords) {
            view(args...) *= x;
        }
    }
};

using Tests = TestParams::tests<1, 2, 3, 4, 5, 6>;
TYPED_TEST_SUITE(FieldSubLayoutTest, Tests);

TYPED_TEST(FieldSubLayoutTest, DeepCopy) {
    auto& field = this->field;

    *field    = 0;
    auto copy = field->deepCopy();
    copy      = copy + 1.;

    auto mirrorA = field->getHostMirror();
    auto mirrorB = copy.getHostMirror();

    Kokkos::deep_copy(mirrorA, field->getView());
    Kokkos::deep_copy(mirrorB, copy.getView());

    nestedViewLoop(mirrorA, field->getNghost(), [&]<typename... Idx>(const Idx... args) {
        assertEqual<typename TestFixture::value_type>(mirrorA(args...) + 1, mirrorB(args...));
    });
}

TYPED_TEST(FieldSubLayoutTest, Sum) {
    using T = typename TestFixture::value_type;

    T val      = 1.0;
    T expected = std::reduce(this->nPoints.begin(), this->nPoints.end(), val, std::multiplies<>{});

    auto& field = this->field;

    *field = val;

    T sum = field->sum();

    assertEqual<T>(expected, sum);
}

TYPED_TEST(FieldSubLayoutTest, Norm1) {
    using T = typename TestFixture::value_type;

    T val      = -1.5;
    T expected = std::reduce(this->nPoints.begin(), this->nPoints.end(), -val, std::multiplies<>{});

    auto& field = this->field;

    *field = val;

    T norm1 = ippl::norm(*field, 1);

    assertEqual<T>(expected, norm1);
}

TYPED_TEST(FieldSubLayoutTest, Norm2) {
    using T = typename TestFixture::value_type;

    T val = 1.5;
    T squared =
        std::reduce(this->nPoints.begin(), this->nPoints.end(), val * val, std::multiplies<>{});

    auto& field = this->field;

    *field = val;

    T norm2 = ippl::norm(*field);

    assertEqual<T>(std::sqrt(squared), norm2);
}

TYPED_TEST(FieldSubLayoutTest, NormInf) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    T val      = 1.;
    T expected = std::accumulate(this->nPoints.begin(), this->nPoints.end(), -val);

    auto& field = this->field;

    const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();

    auto view                     = field->getView();
    const ippl::Vector<T, Dim> dx = field->get_mesh().getMeshSpacing();
    FieldVal<TypeParam> fv(view, lDom, dx);
    Kokkos::parallel_for(
        "Set field", field->template getFieldRangePolicy<typename FieldVal<TypeParam>::Norm>(), fv);

    T normInf = ippl::norm(*field, 0);

    assertEqual<T>(expected, normInf);
}

TYPED_TEST(FieldSubLayoutTest, VolumeIntegral) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    auto& field = this->field;

    /// to avoid error accumulation we increase the tolerance by the number of summands
    std::size_t totalNumberOfPoints = std::accumulate(this->nPoints.begin(), this->nPoints.end(), std::size_t{1}, std::multiplies<>{});

    T tol                         = totalNumberOfPoints * tolerance<T>;

    const ippl::NDIndex<Dim> lDom = field->getLayout().getLocalNDIndex();
    const int shift               = field->getNghost();

    const ippl::Vector<T, Dim> dx = field->get_mesh().getMeshSpacing();
    auto view                     = field->getView();

    FieldVal<TypeParam> fv(view, lDom, dx, shift);
    Kokkos::parallel_for(
        "Set field", field->template getFieldRangePolicy<typename FieldVal<TypeParam>::Integral>(),
        fv);

    ASSERT_NEAR(field->getVolumeIntegral(), 0., tol);
}

TYPED_TEST(FieldSubLayoutTest, VolumeIntegral2) {
    using T = typename TestFixture::value_type;

    auto& field = this->field;

    *field     = 1.;
    T integral = field->getVolumeIntegral();
    T volume   = field->get_mesh().getMeshVolume();

    assertEqual<T>(integral, volume);
}

TYPED_TEST(FieldSubLayoutTest, Grad) {
    auto& field = this->field;

    *field = 1.;

    typename TestFixture::vfield_type vfield(field->get_mesh(), field->getLayout());
    vfield = grad(*field);

    const int shift = vfield.getNghost();
    auto view       = vfield.getView();
    auto mirror     = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, view);

    nestedViewLoop(mirror, shift, [&]<typename... Idx>(const Idx... args) {
        for (size_t d = 0; d < TestFixture::dim; d++) {
            assertEqual<typename TestFixture::value_type>(mirror(args...)[d], 0.);
        }
    });
}

TYPED_TEST(FieldSubLayoutTest, Div) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    auto& field = this->field;

    typename TestFixture::vfield_type vfield(field->get_mesh(), field->getLayout());
    auto view        = vfield.getView();
    const int vshift = vfield.getNghost();

    const ippl::NDIndex<Dim> lDom = vfield.getLayout().getLocalNDIndex();

    const ippl::Vector<T, Dim> dx = vfield.get_mesh().getMeshSpacing();
    VFieldVal<TypeParam> fv(view, lDom, dx, vshift);
    Kokkos::parallel_for("Set field", vfield.getFieldRangePolicy(vshift), fv);

    *field = div(vfield);

    const int shift = field->getNghost();
    auto mirror     = Kokkos::create_mirror_view(field->getView());
    Kokkos::deep_copy(mirror, field->getView());

    nestedViewLoop(mirror, shift, [&]<typename... Idx>(const Idx... args) {
        assertEqual<T>(mirror(args...), Dim);
    });
}

TYPED_TEST(FieldSubLayoutTest, Curl) {
    constexpr unsigned Dim = TestFixture::dim;
    // Restrict to 3D case for now
    if constexpr (Dim == 3) {
        using T = typename TestFixture::value_type;

        using vfield_type = typename TestFixture::vfield_type;

        auto& mesh   = this->mesh;
        auto& layout = this->layout;

        vfield_type vfield(*mesh, *layout);
        const int nghost = vfield.getNghost();
        auto view_field  = vfield.getView();

        ippl::NDIndex<Dim> lDom     = layout->getLocalNDIndex();
        ippl::Vector<T, Dim> hx     = mesh->getMeshSpacing();
        ippl::Vector<T, Dim> origin = mesh->getOrigin();

        auto mirror = Kokkos::create_mirror_view(view_field);
        Kokkos::deep_copy(mirror, view_field);

        for (unsigned int gd = 0; gd < Dim; ++gd) {
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
        auto view       = result.getView();
        mirror          = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(mirror, view);

        for (size_t i = shift; i < mirror.extent(0) - shift; ++i) {
            for (size_t j = shift; j < mirror.extent(1) - shift; ++j) {
                for (size_t k = shift; k < mirror.extent(2) - shift; ++k) {
                    for (size_t d = 0; d < Dim; ++d) {
                        assertEqual<T>(mirror(i, j, k)[d], 0.);
                    }
                }
            }
        }
    }
}

TYPED_TEST(FieldSubLayoutTest, Hessian) {
    using T                = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    auto& field  = this->field;
    auto& mesh   = this->mesh;
    auto& layout = this->layout;

    typedef ippl::Vector<T, Dim> Vector_t;
    typedef ippl::Field<ippl::Vector<Vector_t, Dim>, Dim, typename TestFixture::mesh_type,
                        typename TestFixture::centering_type, typename TestFixture::exec_space>
        MField_t;

    int nghost      = field->getNghost();
    auto view_field = field->getView();

    ippl::NDIndex<Dim> lDom = layout->getLocalNDIndex();
    Vector_t hx             = mesh->getMeshSpacing();
    Vector_t origin         = mesh->getOrigin();

    FieldVal<TypeParam> fv(view_field, lDom, hx, nghost);
    Kokkos::parallel_for(
        "Set field",
        field->template getFieldRangePolicy<typename FieldVal<TypeParam>::Hessian>(nghost), fv);

    MField_t result(*mesh, *layout);
    result = hess(*field);

    nghost             = result.getNghost();
    auto view_result   = result.getView();
    auto mirror_result = Kokkos::create_mirror_view(view_result);
    Kokkos::deep_copy(mirror_result, view_result);

    constexpr T tol = tolerance<T>;
    nestedViewLoop(mirror_result, nghost, [&]<typename... Idx>(const Idx... args) {
        T det = 0;
        for (unsigned d = 0; d < Dim; d++) {
            det += mirror_result(args...)[d][d];
        }
        ASSERT_NEAR(det, 0., tol);
    });
}

TYPED_TEST(FieldSubLayoutTest, Laplace) {
    auto& mesh   = this->mesh;
    auto& layout = this->layout;
    auto& field  = this->field;

    using field_type     = typename TestFixture::field_type;
    using T              = typename TestFixture::value_type;
    constexpr size_t Dim = TestFixture::dim;

    field_type laplacian(*mesh, *layout);

    const int nghost = field->getNghost();

    using bc_type = ippl::BConds<field_type, Dim>;
    bc_type bcField;
    for (size_t i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ConstantFace<field_type>>(i, 1);
    }

    field->setFieldBC(bcField);
    laplacian.setFieldBC(bcField);

    *field    = 1;
    laplacian = ippl::laplace(*field);

    auto mirror = laplacian.getHostMirror();
    Kokkos::deep_copy(mirror, laplacian.getView());
    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        assertEqual<T>(mirror(args...), 0.);
    });
}
TYPED_TEST(FieldSubLayoutTest, LowerLaplace) {
    auto& mesh   = this->mesh;
    auto& layout = this->layout;
    auto& field  = this->field;

    using field_type     = typename TestFixture::field_type;
    using T              = typename TestFixture::value_type;
    constexpr size_t Dim = TestFixture::dim;

    field_type laplacian(*mesh, *layout);
    field_type lower_laplacian(*mesh, *layout);
    field_type upper_laplacian(*mesh, *layout);
    field_type diagonal_laplacian(*mesh, *layout);

    const int nghost = field->getNghost();

    using bc_type = ippl::BConds<field_type, Dim>;
    bc_type bcField;
    for (size_t i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
    }

    field->setFieldBC(bcField);

    lower_laplacian.setFieldBC(bcField);
    upper_laplacian.setFieldBC(bcField);
    diagonal_laplacian.setFieldBC(bcField);
    laplacian.setFieldBC(bcField);

    double diagonal_factor = 0;
    for (unsigned d = 0; d < Dim; ++d) {
        diagonal_factor += 2.0 / std::pow(mesh->getMeshSpacing(d), 2);
    }

    *field             = 1;
    lower_laplacian    = ippl::lower_laplace(*field);
    upper_laplacian    = ippl::upper_laplace(*field);
    diagonal_laplacian = -diagonal_factor * (*field);
    laplacian          = lower_laplacian + diagonal_laplacian + upper_laplacian;

    auto mirror = laplacian.getHostMirror();
    Kokkos::deep_copy(mirror, laplacian.getView());
    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        assertEqual<T>(mirror(args...), 0.);
    });
}

TYPED_TEST(FieldSubLayoutTest, UpperAndLowerLaplace) {
    auto& mesh   = this->mesh;
    auto& layout = this->layout;
    auto& field  = this->field;

    using field_type     = typename TestFixture::field_type;
    using T              = typename TestFixture::value_type;
    constexpr size_t Dim = TestFixture::dim;

    field_type laplacian(*mesh, *layout);
    field_type upper_and_lower_laplacian(*mesh, *layout);
    field_type diagonal_laplacian(*mesh, *layout);

    const int nghost = field->getNghost();

    using bc_type = ippl::BConds<field_type, Dim>;
    bc_type bcField;
    for (size_t i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
    }

    field->setFieldBC(bcField);

    upper_and_lower_laplacian.setFieldBC(bcField);
    diagonal_laplacian.setFieldBC(bcField);
    laplacian.setFieldBC(bcField);

    double diagonal_factor = 0;
    for (unsigned d = 0; d < Dim; ++d) {
        diagonal_factor += 2.0 / std::pow(mesh->getMeshSpacing(d), 2);
    }

    *field                    = 1;
    upper_and_lower_laplacian = ippl::upper_and_lower_laplace(*field);
    diagonal_laplacian        = -diagonal_factor * (*field);
    laplacian                 = upper_and_lower_laplacian + diagonal_laplacian;

    auto mirror = laplacian.getHostMirror();
    Kokkos::deep_copy(mirror, laplacian.getView());
    nestedViewLoop(mirror, nghost, [&]<typename... Idx>(const Idx... args) {
        assertEqual<T>(mirror(args...), 0.);
    });
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
