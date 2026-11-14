// Unit tests: optional particle transforms have matching assembly and data blocks.
// The capture backend checks mixed channels, empty particles, transform updates,
// late registration and channel rebuilds, using two MPI ranks by default.
// Particle and field data use the default Kokkos execution/memory space.

#include "Stream/InSitu/CatalystAdaptor.h"
#include "SnapshotTestUtils.h"

namespace {
using Vector = ippl::Vector<double, 3>;
using Mesh = ippl::UniformCartesian<double, 3>;
using Layout = ippl::ParticleSpatialLayout<double, 3, Mesh>;
using ScalarField = ippl::Field<double, 3, Mesh, Mesh::DefaultCentering>;

struct Particles {
    ippl::NDIndex<3> owned_m{ippl::Index(8), ippl::Index(8), ippl::Index(8)};
    ippl::FieldLayout<3> fieldLayout_m{*ippl::Comm, owned_m, {true, true, true}, true};
    Mesh mesh_m{owned_m, Vector(1.), Vector(0.)};
    Layout layout_m{fieldLayout_m, mesh_m};
    ippl::ParticleBase<Layout> bunch_m{layout_m};

    explicit Particles(size_t count = 2) {
        bunch_m.create(count);
        if (count > 0) bunch_m.R = Vector(0.25, 0.5, 0.75);
    }
};

ippl::TransformData makeTransform(double scale) {
    ippl::TransformData transform;
    transform.origin = Vector(scale, 2. * scale, 3. * scale);
    transform.rows = {Vector(0., -1., 0.), Vector(1., 0., 0.), Vector(0., 0., 1.)};
    transform.invOrigin = Vector(2. * scale, -scale, -3. * scale);
    transform.invRows = {Vector(0., 1., 0.), Vector(-1., 0., 0.), Vector(0., 0., 1.)};
    return transform;
}

void expectTransform(conduit_cpp::Node channel, const ippl::TransformData& transform) {
    ASSERT_TRUE(channel.has_path("assembly/transform"));
    ASSERT_TRUE(channel.has_path("data/transform/coordsets/tf_coords/type"));
    ASSERT_TRUE(channel.has_path("data/transform/topologies/tf_topo/elements/connectivity"));
    const char* names[] = {"BunchOrigin", "BunchRotRow0", "BunchRotRow1", "BunchRotRow2",
                           "BunchInvOrigin", "BunchInvRotRow0", "BunchInvRotRow1",
                           "BunchInvRotRow2"};
    const Vector values[] = {transform.origin, transform.rows[0], transform.rows[1],
                              transform.rows[2], transform.invOrigin, transform.invRows[0],
                              transform.invRows[1], transform.invRows[2]};
    const char* axes[] = {"x", "y", "z"};
    for (unsigned i = 0; i < 8; ++i) {
        const std::string field = std::string("data/transform/fields/") + names[i];
        ASSERT_TRUE(channel.has_path(field + "/association"));
        ASSERT_TRUE(channel.has_path(field + "/topology"));
        for (unsigned d = 0; d < 3; ++d) {
            const std::string path = field + "/values/" + axes[d];
            ASSERT_TRUE(channel.has_path(path));
            snapshot_test::expectValues(channel[path], {values[i][d]});
        }
    }
}
}  // namespace

// An untransformed channel must omit both transform paths, including an initially
// empty container and a subsequent particle-count change that rebuilds the channel.
TEST(CatalystParticleTransforms, NoTransformForEmptyAndGrowingParticles) {
    Particles particles(0);
    ippl::CatalystAdaptor adaptor;
    adaptor.addParticleChannel("plain", particles.bunch_m);
    adaptor.Initialize();
    for (int cycle = 0; cycle < 2; ++cycle) {
        if (cycle == 1) {
            particles.bunch_m.create(2);
            particles.bunch_m.R = Vector(0.25, 0.5, 0.75);
        }
        adaptor.Execute(cycle, 0.1 * cycle);
        auto root = snapshot_test::captured();
        auto channel = root["catalyst/channels/plain"];
        EXPECT_FALSE(channel.has_path("assembly/transform"));
        EXPECT_FALSE(channel.has_path("data/transform"));
        EXPECT_TRUE(channel.has_path("data/particles"));
        EXPECT_TRUE(channel.has_path("data/boundingbox"));
    }
    adaptor.Finalize();
}

// A transform belongs only to its registered channel. Origin, rotation and inverse
// values must update on later exports without changing the exported particle positions.
TEST(CatalystParticleTransforms, MixedChannelsKeepTransformValuesCurrent) {
    Particles particles;
    ippl::CatalystAdaptor adaptor;
    auto beam = adaptor.addParticleChannel("beam", particles.bunch_m);
    adaptor.addParticleChannel("plain", particles.bunch_m);
    beam.setTransform(makeTransform(1.));
    adaptor.Initialize();
    for (int cycle = 0; cycle < 2; ++cycle) {
        const auto transform = makeTransform(cycle + 1.);
        beam.setTransform(transform);
        adaptor.Execute(cycle, 0.1 * cycle);
        auto root = snapshot_test::captured();
        expectTransform(root["catalyst/channels/beam"], transform);
        auto plain = root["catalyst/channels/plain"];
        EXPECT_FALSE(plain.has_path("assembly/transform"));
        EXPECT_FALSE(plain.has_path("data/transform"));
        snapshot_test::expectValues(
            root["catalyst/channels/beam/data/particles/coordsets/p_explicit_coords/values/x"],
            {0.25, 0.25});
    }
    adaptor.Finalize();
}

// Adding the first transform after an export must create a complete mesh block.
// Rebuilding its parent, then changing particle count, must preserve a nested field snapshot.
TEST(CatalystParticleTransforms, LateTransformAndRebuildPreserveNestedMesh) {
    Particles particles;
    ScalarField field(particles.mesh_m, particles.fieldLayout_m);
    ippl::CatalystAdaptor adaptor;
    auto beam = adaptor.addParticleChannel("beam", particles.bunch_m);
    adaptor.addMeshBlock("beam", "mesh", particles.mesh_m, particles.fieldLayout_m)
        .addArray("density", field, ippl::UpdatePolicy::Manual);
    adaptor.Initialize();
    field = -1.;
    adaptor.Execute(0, 0.);
    auto initial = snapshot_test::captured();
    EXPECT_FALSE(initial.has_path("catalyst/channels/beam/assembly/transform"));

    for (int cycle = 1; cycle < 3; ++cycle) {
        if (cycle == 2) {
            particles.bunch_m.create(1);
            particles.bunch_m.R = Vector(0.25, 0.5, 0.75);
        }
        const auto transform = makeTransform(cycle);
        beam.setTransform(transform);
        field = double(cycle);
        adaptor.refresh("density");
        field = -double(cycle);
        adaptor.Execute(cycle, 0.1 * cycle);
        auto root = snapshot_test::captured();
        expectTransform(root["catalyst/channels/beam"], transform);
        const auto local = particles.fieldLayout_m.getLocalNDIndex();
        const auto count = local[0].length() * local[1].length() * local[2].length();
        snapshot_test::expectValues(
            root["catalyst/channels/beam/data/mesh/fields/density/values"],
            std::vector<double>(count, double(cycle)));
    }
    adaptor.Finalize();
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
