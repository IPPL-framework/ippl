//
// Unit tests for gather/scatter functionality (multi-rank compatible)
//   Tests gather with addToAttribute = false and true,
//   scatter with a custom range policy,
//   and scatter with a custom hash_type.
//
// These tests extend the functionality tests from the original
// TestHashedScatter.cpp and TestGather.cpp examples.
//

#include "Ippl.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

// A helper needed to reduce over a hash_type.
// This is needed, since Kokkos kernels apparently
// cannot be called inside a TYPED_TEST on device.
struct ComputeTotalChargeLambda {
    Kokkos::View<double*> viewQ;
    Kokkos::View<int*> hash;

    ComputeTotalChargeLambda(Kokkos::View<double*> viewQ_, Kokkos::View<int*> hash_) 
        : viewQ(viewQ_), hash(hash_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i, double& val) const {
        val += viewQ(hash(i));
    }
};

// A simple bunch_type holding a charge attribute 
template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    Bunch(PLayout& playout)
        : ippl::ParticleBase<PLayout>(playout) {
        this->addAttribute(Q);
    }
    ~Bunch() = default;

    typedef ippl::ParticleAttrib<double, typename PLayout::position_execution_space> charge_container_type;
    charge_container_type Q;
};

template <typename>
class GatherScatterTest;

template <typename T, typename ExecSpace, unsigned Dim>
class GatherScatterTest<Parameters<T, ExecSpace, Rank<Dim>>> : public ::testing::Test {
public:
    using scalar_type    = T;
    using exec_space     = ExecSpace;
    static const unsigned dim = Dim;
    using flayout_type   = ippl::FieldLayout<Dim>;
    using mesh_type      = ippl::UniformCartesian<T, Dim>;
    using playout_type   = ippl::ParticleSpatialLayout<T, Dim, mesh_type, ExecSpace>;
    using bunch_type     = Bunch<playout_type>;

    // Domain parameters: use a high resolution grid so that cells are small.
    std::array<size_t, Dim> nPoints;
    std::array<T, Dim> domain;
    flayout_type layout;
    mesh_type mesh;
    std::shared_ptr<playout_type> playout;    
    std::shared_ptr<bunch_type> bunch;

    // Particle counts for the tests.
    size_t nGather = 10;              // for gather test: local particles per rank
    size_t nScatter = static_cast<unsigned int>(std::pow(64, Dim));  // for scatter tests

    // Store cell sizes (hx) for use in generating positions.
    T hx[Dim];

    GatherScatterTest() { }

    void SetUp() override {
        // Use a high-resolution grid (e.g. 512 cells per dimension)
        size_t gridPoints = 128;
        for (size_t d = 0; d < Dim; d++) {
            nPoints[d] = gridPoints;
            domain[d]  = 1.0;
        }
        std::array<ippl::Index, Dim> owned;
        for (size_t d = 0; d < Dim; d++) {
            owned[d] = ippl::Index(nPoints[d]);
        }
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        auto owned_tu = std::make_from_tuple<ippl::NDIndex<Dim>>(owned);
        layout = flayout_type(MPI_COMM_WORLD, owned_tu, isParallel);

        ippl::Vector<T, Dim> hx_vec;
        ippl::Vector<T, Dim> origin;
        for (size_t d = 0; d < Dim; d++) {
            hx_vec[d] = domain[d] / nPoints[d];
            hx[d] = hx_vec[d]; // store cell size for distribution
            origin[d] = 0;
        }
        mesh    = mesh_type(owned_tu, hx_vec, origin);
        playout = std::make_shared<playout_type>(layout, mesh);
        bunch   = std::make_shared<bunch_type>(*playout);

        // Set periodic boundary conditions.
        bunch->setParticleBC(ippl::BC::PERIODIC);
    }

    // Fill particle positions with random numbers in [hx/2, 1-hx/2] to ensure interior points.
    void fillRandomPositions(size_t nParticles) {
        bunch->create(nParticles);
        std::mt19937_64 eng(ippl::Comm->rank());
        std::uniform_real_distribution<T> unif(hx[0] / 2, 1 - (hx[0] / 2));
        auto R_host = bunch->R.getHostMirror();
        for (size_t i = 0; i < nParticles; ++i) {
            ippl::Vector<T, Dim> r;
            for (size_t d = 0; d < Dim; d++) {
                r[d] = unif(eng);
            }
            R_host(i) = r;
        }
        Kokkos::deep_copy(bunch->R.getView(), R_host);
        ippl::Comm->barrier();
        Kokkos::fence();
        bunch->update();
    }

    // Fill the Q attribute with a constant value using host mirror and deep copy.
    void fillAttributeQ(T value) {
        auto Q_host = bunch->Q.getHostMirror();
        for (size_t i = 0; i < Q_host.size(); ++i) {
            Q_host(i) = value;
        }
        Kokkos::deep_copy(bunch->Q.getView(), Q_host);
        ippl::Comm->barrier();
    }
};

using TestTypes = ::testing::Types<
    Parameters<double, Kokkos::DefaultExecutionSpace, Rank<1>>,
    Parameters<double, Kokkos::DefaultExecutionSpace, Rank<2>>,
    Parameters<double, Kokkos::DefaultExecutionSpace, Rank<3>>//,
    //Parameters<double, Kokkos::DefaultExecutionSpace, Rank<4>>,
    //Parameters<double, Kokkos::DefaultExecutionSpace, Rank<5>>,
    //Parameters<double, Kokkos::DefaultExecutionSpace, Rank<6>>
>;
TYPED_TEST_SUITE(GatherScatterTest, TestTypes);

//
// GatherTest: 
// First, set each local Q to 10.0.
// Then, call gather with addToAttribute = false so that Q becomes 1.0 (should replace value with 1.0).
// If Q != 0, then the values were 1. not replaced and 2. not correctly gathered from the field.
// Note: for a constant field, there should not be an error during linear interpolation.
//
TYPED_TEST(GatherScatterTest, GatherTestReplace) {
    const size_t n = this->nGather;
    this->fillRandomPositions(n);
    this->fillAttributeQ(10.0);

    using Mesh_t   = typename TestFixture::mesh_type;
    using FieldType = ippl::Field<typename TestFixture::scalar_type, TestFixture::dim, Mesh_t, typename Mesh_t::DefaultCentering, typename TestFixture::exec_space>;
    FieldType field;
    field.initialize(this->mesh, this->layout);
    field = 1.0;

    // First gather call (no accumulation, replace attributes)
    gather(this->bunch->Q, field, this->bunch->R);

    // Check all charges
    auto Q_host = this->bunch->Q.getHostMirror();
    Kokkos::deep_copy(Q_host, this->bunch->Q.getView());
    for (size_t i = 0; i < this->bunch->getLocalNum(); ++i) {
        ASSERT_NEAR(Q_host(i), 1.0, 1e-6);
    }
}

//
// GatherTest: 
// First, set each local Q to 1.0.
// Then, call gather with addToAttribute = true so that Q becomes 2.0 (should add 1.0 per particle).
//
TYPED_TEST(GatherScatterTest, GatherTestIncrement) {
    const size_t n = this->nGather;
    this->fillRandomPositions(n);
    this->fillAttributeQ(1.0);

    using Mesh_t   = typename TestFixture::mesh_type;
    using FieldType = ippl::Field<typename TestFixture::scalar_type, TestFixture::dim, Mesh_t, typename Mesh_t::DefaultCentering, typename TestFixture::exec_space>;
    FieldType field;
    field.initialize(this->mesh, this->layout);
    field = 1.0;

    // Second gather call with addToAttribute=true should add another 1.0.
    gather(this->bunch->Q, field, this->bunch->R, true);

    // Check all charges
    auto Q_host = this->bunch->Q.getHostMirror();
    Kokkos::deep_copy(Q_host, this->bunch->Q.getView());
    for (unsigned int i = 0; i < this->bunch->getLocalNum(); ++i) {
        ASSERT_NEAR(Q_host(i), 2.0, 1e-6);
    }
}

//
// ScatterSimpleTest:
// Set Q = 1.0 for all particles and scatter them to the field.
// Then compare the total charge in the field to the total charge from the particles.
// (adapted from test/particle/TestScatter.cpp)
//
TYPED_TEST(GatherScatterTest, ScatterSimpleTest) {
    const unsigned int n = this->nScatter;
    this->fillRandomPositions(n);
    this->fillAttributeQ(1.0);

    // Create and initialize a field/mesh.
    using Mesh_t    = typename TestFixture::mesh_type;
    using FieldType = ippl::Field<typename TestFixture::scalar_type, TestFixture::dim, Mesh_t, typename Mesh_t::DefaultCentering>;
    FieldType field;
    field.initialize(this->mesh, this->layout);
    
    field = 0.0;

    // Perform the simple scatter operation (extended functionality is tested below).
    scatter(this->bunch->Q, field, this->bunch->R);

    // Compute the total charge in the field and from the particles.
    double total_field = field.sum();
    double total_particles = this->bunch->Q.sum();

    // Check that the scattered field conserves charge.
    ASSERT_NEAR(total_field, total_particles, 1e-6);
}


//
// ScatterCustomRangeTest: 
// Set Q = 1.0 for all particles and scatter only a subset defined by a custom range policy.
// Then compare the total charge in the field to the expected value.
//
TYPED_TEST(GatherScatterTest, ScatterCustomRangeTest) {
    const size_t n = this->nScatter;
    if(n % ippl::Comm->size() != 0) {
        GTEST_SKIP() << "nScatter not divisible by number of ranks.";
    }
    this->fillRandomPositions(n);
    this->fillAttributeQ(1.0);

    using Mesh_t   = typename TestFixture::mesh_type;
    using FieldType = ippl::Field<typename TestFixture::scalar_type, TestFixture::dim, Mesh_t, typename Mesh_t::DefaultCentering>;
    FieldType field;
    field.initialize(this->mesh, this->layout);
    field = 0.0;

    size_t rank = ippl::Comm->rank();
    size_t nLoc = this->bunch->getLocalNum();
    size_t NScattered = nLoc / 2 + rank;

    double Q_total = 1.0 * NScattered;
    ippl::Comm->allreduce(Q_total, 1, std::plus<double>());

    Kokkos::RangePolicy<typename TestFixture::exec_space> policy(0, NScattered);
    scatter(this->bunch->Q, field, this->bunch->R, policy);

    double Total_charge_field = field.sum();
    ASSERT_NEAR(Q_total, Total_charge_field, 1e-6);
}

//
// ScatterCustomHashTest: 
// Assign random charges (in [0.5, 1.5]), create and shuffle an index array,
// use it as a custom hash, scatter the first NScattered particles accordingly,
// and compare the field’s total charge to the expected total.
//
TYPED_TEST(GatherScatterTest, ScatterCustomHashTest) {
    const size_t n = this->nScatter / ippl::Comm->size();
    if (this->nScatter % ippl::Comm->size() > 0) {
        GTEST_SKIP() << "nScatter not divisible by number of ranks.";
    }
    this->fillRandomPositions(n);
    
    size_t rank = ippl::Comm->rank();
    size_t nLoc = this->bunch->getLocalNum(); // since update() might change number of particles 
    size_t NScattered = nLoc / 2 + rank; // can be anything

    // Assign random charges to particles
    std::mt19937_64 eng(42);
    std::uniform_real_distribution<typename TestFixture::scalar_type> unif_charge(0.5, 1.5);
    auto Q_host = this->bunch->Q.getHostMirror();
    for (size_t i = 0; i < n; ++i) {
        Q_host(i) = unif_charge(eng);
    }
    Kokkos::deep_copy(this->bunch->Q.getView(), Q_host);

    // Create and initialize a field/mesh.
    using Mesh_t   = typename TestFixture::mesh_type;
    using FieldType = ippl::Field<typename TestFixture::scalar_type, TestFixture::dim, Mesh_t, typename Mesh_t::DefaultCentering>;
    FieldType field;
    field.initialize(this->mesh, this->layout);
    field = 0.0;

    // Create a custom hash using a shuffled index array
    using hash_type = typename TestFixture::bunch_type::charge_container_type::hash_type;
    hash_type hash("indexArray", nLoc);
    std::vector<int> host_indices(nLoc);
    std::iota(host_indices.begin(), host_indices.end(), 0);
    std::shuffle(host_indices.begin(), host_indices.end(), eng);

    // Copy shuffled index array to the hash_type
    auto hash_host = Kokkos::create_mirror_view(hash);
    for (size_t i = 0; i < nLoc; ++i) {
        hash_host(i) = host_indices[i];
    }
    Kokkos::deep_copy(hash, hash_host);

    // First compute the total charge of the first NScattered particles as determined by the hash map
    double Q_total = 0.0;
    auto viewQ = this->bunch->Q.getView();

    ComputeTotalChargeLambda lambda(viewQ, hash);
    Kokkos::parallel_reduce("computeTotalCharge", 
        Kokkos::RangePolicy<typename TestFixture::exec_space>(0, NScattered),
        lambda, Q_total);
    /*Kokkos::parallel_reduce("computeTotalCharge", 
        Kokkos::RangePolicy<typename TestFixture::exec_space>(0, NScattered),
        KOKKOS_LAMBDA(const size_t i, double& val) {
            val += viewQ(hash(i));
        }, Q_total);*/
    ippl::Comm->allreduce(Q_total, 1, std::plus<double>());

    // Scatter the first NScattered particles using the custom hash
    Kokkos::RangePolicy<typename TestFixture::exec_space> policy(0, NScattered);
    scatter(this->bunch->Q, field, this->bunch->R, policy, hash);

    // Check the total charge in the field and compare to the expected total
    double Total_charge_field = field.sum();
    ASSERT_NEAR(Q_total, Total_charge_field, 1e-6);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int result = 1;
    {
        ::testing::InitGoogleTest(&argc, argv);
        result = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return result;
}
