#include "Ippl.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
    }
    ~Bunch() {}
    typedef ippl::ParticleAttrib<double> charge_container_type;
    charge_container_type Q;
};

template <typename>
class AssembleCurrentYeeTest;

template <typename T, unsigned Dim>
class AssembleCurrentYeeTest<Parameters<T, Rank<Dim>>> : public ::testing::Test {
public:
    using value_type   = T;
    static constexpr unsigned dim = Dim;

    using Mesh_t       = ippl::UniformCartesian<T, Dim>;
    using Centering_t  = typename Mesh_t::DefaultCentering;
    using JField_t     = ippl::Field<ippl::Vector<T, Dim>, Dim, Mesh_t, Centering_t>;
    using Layout_t     = ippl::FieldLayout<Dim>;

    using playout_t    = ippl::ParticleSpatialLayout<T, Dim>;
    using bunch_t      = Bunch<playout_t>;

    static ippl::NDIndex<Dim> make_owned_nd(int nx) {
        ippl::Index I0(nx);
        if constexpr (Dim == 1)      return ippl::NDIndex<1>(I0);
        else if constexpr (Dim == 2) return ippl::NDIndex<2>(I0, I0);
        else                         return ippl::NDIndex<3>(I0, I0, I0);
    }

    static Layout_t make_layout(const ippl::NDIndex<Dim>& owned) {
        std::array<bool, Dim> par{}; par.fill(true);
        return ippl::FieldLayout<Dim>(MPI_COMM_WORLD, owned, par);
    }

    static Mesh_t make_mesh(const ippl::NDIndex<Dim>& owned,
                            const ippl::Vector<T, Dim>& h,
                            const ippl::Vector<T, Dim>& origin) {
        return Mesh_t(owned, h, origin);
    }
};

using Precisions = TestParams::Precisions;
using Ranks      = TestParams::Ranks<2, 3>;
using Tests      = TestForTypes<CreateCombinations<Precisions, Ranks>::type>::type;

TYPED_TEST_SUITE(AssembleCurrentYeeTest, Tests);

TYPED_TEST(AssembleCurrentYeeTest, DiagonalPath_ThreeCells_ExactValues) {
    using T = typename TestFixture::value_type;
    constexpr unsigned Dim = TestFixture::dim;

    if constexpr (Dim != 2) {
        GTEST_SKIP() << "Exact value check only implemented for 2D";
    } else {
        using bunch_t   = typename TestFixture::bunch_t;
        using playout_t = typename TestFixture::playout_t;
        using JField_t  = typename TestFixture::JField_t;

        int nx = 4;
        ippl::Vector<T, Dim> origin(0.0);
        ippl::Vector<T, Dim> h(1.0);

        auto owned  = TestFixture::make_owned_nd(nx);
        auto layout = TestFixture::make_layout(owned);
        auto mesh   = TestFixture::make_mesh(owned, h, origin);

        JField_t J_field(mesh, layout);
        J_field = T(0);

        // path: (0.75, 0.50) -> (1.50, 1.25), same as FEM test
        playout_t playout(layout, mesh);
        bunch_t bunch(playout);
        bunch.create(1);
        {
            auto R_host = bunch.R.getHostMirror();
            auto Q_host = bunch.Q.getHostMirror();
            R_host(0)[0] = T(0.75);
            R_host(0)[1] = T(0.50);
            Q_host(0) = T(1.0);
            Kokkos::deep_copy(bunch.R.getView(), R_host);
            Kokkos::deep_copy(bunch.Q.getView(), Q_host);
            bunch.update();
        }

        bunch_t bunch_next(playout);
        bunch_next.create(1);
        {
            auto Rn_host = bunch_next.R.getHostMirror();
            Rn_host(0)[0] = T(1.50);
            Rn_host(0)[1] = T(1.25);
            Kokkos::deep_copy(bunch_next.R.getView(), Rn_host);
            bunch_next.update();
        }

        auto policy = Kokkos::RangePolicy<>(0, bunch.getLocalNum());
        T dt = T(1.0);
        ippl::assemble_current_yee(mesh, bunch.Q, bunch.R, bunch_next.R,
                                   J_field, policy, dt);
        J_field.accumulateHalo();

        auto ldom  = J_field.getLayout().getLocalNDIndex();
        int nghost = J_field.getNghost();

        auto view      = J_field.getView();
        auto view_host = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(view_host, view);
        Kokkos::fence();

        const T tol = std::numeric_limits<T>::epsilon() * T(100);

        // Check components at grid point (gi, gj) if owned by this rank
        auto check = [&](int gi, int gj, unsigned c, double expected) {
            if (gi >= ldom.first()[0] && gi <= ldom.last()[0] &&
                gj >= ldom.first()[1] && gj <= ldom.last()[1]) {
                int li = gi - ldom.first()[0] + nghost;
                int lj = gj - ldom.first()[1] + nghost;
                EXPECT_NEAR(static_cast<double>(view_host(li, lj)[c]),
                            expected, static_cast<double>(tol));
            }
        };

        check(0, 0, 0, 0.09375);
        check(0, 0, 1, 0.03125);
        check(0, 1, 0, 0.15625);
        check(1, 0, 0, 0.03125);
        check(1, 0, 1, 0.4375);   
        check(1, 1, 0, 0.4375);  
        check(1, 1, 1, 0.15625);
        check(1, 2, 0, 0.03125);
        check(2, 0, 1, 0.03125);
        check(2, 1, 1, 0.09375);
        
    }
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
