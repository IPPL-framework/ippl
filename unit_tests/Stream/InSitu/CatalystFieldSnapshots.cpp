// Unit test: Catalyst field snapshots survive source reuse and channel rebuilds.
// Inspects the capture backend's exact payload with two MPI ranks by default.
// Scalar/vector sources use the default Kokkos execution/memory space, with an
// additional HostSpace scalar. Captures must own their values independently.

#include "Stream/InSitu/CatalystAdaptor.h"
#include "SnapshotTestUtils.h"

namespace {
using Vector = ippl::Vector<double, 3>;
using Mesh = ippl::UniformCartesian<double, 3>;
using ScalarField = ippl::Field<double, 3, Mesh, Mesh::DefaultCentering>;
using VectorField = ippl::Field<Vector, 3, Mesh, Mesh::DefaultCentering>;
using HostField = ippl::Field<double, 3, Mesh, Mesh::DefaultCentering, Kokkos::HostSpace>;

struct Domain {
    ippl::NDIndex<3> owned{ippl::Index(8), ippl::Index(8), ippl::Index(8)};
    ippl::FieldLayout<3> layout{*ippl::Comm, owned, {true, true, true}, true};
    Mesh mesh{owned, Vector(1.), Vector(0.)};

    void resize(unsigned n) {
        owned = ippl::NDIndex<3>{ippl::Index(n), ippl::Index(n), ippl::Index(n)};
        layout.initialize(owned, {true, true, true}, true);
        mesh.initialize(owned, Vector(1.), Vector(0.));
    }
};

void expectConstant(conduit_cpp::Node root, const std::string& path,
                    const Domain& domain, double expected) {
    const auto local = domain.layout.getLocalNDIndex();
    const auto count = local[0].length() * local[1].length() * local[2].length();
    snapshot_test::expectValues(root[path], std::vector<double>(count, expected));
}

const std::string VALUES = "catalyst/channels/grid/data/fields/density/values";
}  // namespace

// A legacy rememberNow call before first Execute must capture immediately. A
// later Execute without a new capture resumes automatic updates; last capture wins.
TEST(CatalystFieldSnapshots, FirstCaptureAndAutomaticUpdates) {
    Domain domain;
    ScalarField field(domain.mesh, domain.layout);
    ippl::CatalystAdaptor adaptor;
    adaptor.addMeshChannel("grid", domain.mesh, domain.layout).addArray("density", field);
    adaptor.Initialize();
    const double rank = ippl::Comm->rank();
    field = rank + 2.;
    adaptor.rememberNow("density");
    field = rank + 9.;
    adaptor.Execute(1, 0.1);
    expectConstant(snapshot_test::captured(), VALUES, domain, rank + 2.);

    field = rank + 3.;
    adaptor.Execute(2, 0.2);
    expectConstant(snapshot_test::captured(), VALUES, domain, rank + 3.);

    field = rank + 4.;
    adaptor.refresh("density");
    field = rank + 5.;
    adaptor.refresh("density");
    field = rank + 10.;
    adaptor.Execute(3, 0.3);
    expectConstant(snapshot_test::captured(), VALUES, domain, rank + 5.);
    adaptor.Finalize();
}

// Manual captures persist through later exports. Automatic vector arrays update
// independently, while a Static HostSpace array retains its initial values.
TEST(CatalystFieldSnapshots, IndependentUpdatePolicies) {
    Domain domain;
    ScalarField density(domain.mesh, domain.layout);
    VectorField electric(domain.mesh, domain.layout);
    HostField fixed(domain.mesh, domain.layout);
    ippl::CatalystAdaptor adaptor;
    adaptor.addMeshChannel("grid", domain.mesh, domain.layout)
        .addArray("density", density, ippl::UpdatePolicy::Manual)
        .addArray("electric", electric)
        .addArray("fixed", fixed, ippl::UpdatePolicy::Static);
    adaptor.Initialize();
    density = 2.;
    adaptor.refresh("density");
    density = 9.;
    electric = Vector(1., 2., 3.);
    fixed = 4.;
    adaptor.Execute(1, 0.1);
    expectConstant(snapshot_test::captured(), VALUES, domain, 2.);

    density = 10.;
    electric = Vector(5., 6., 7.);
    fixed = 8.;
    adaptor.Execute(2, 0.2);
    auto root = snapshot_test::captured();
    expectConstant(root, VALUES, domain, 2.);
    expectConstant(root, "catalyst/channels/grid/data/fields/fixed/values", domain, 4.);
    for (unsigned d = 0; d < 3; ++d) {
        const char* axes[] = {"x", "y", "z"};
        expectConstant(root, std::string("catalyst/channels/grid/data/fields/electric/values/")
                                + axes[d], domain, 5. + d);
    }

    density = 11.;
    adaptor.refresh("density");
    density = 12.;
    adaptor.Execute(3, 0.3);
    expectConstant(snapshot_test::captured(), VALUES, domain, 11.);
    adaptor.Finalize();
}

// Vector captures must preserve every component, including the first capture
// before initialization has allocated a host buffer.
TEST(CatalystFieldSnapshots, VectorCaptureBeforeFirstExecute) {
    Domain domain;
    VectorField field(domain.mesh, domain.layout);
    ippl::CatalystAdaptor adaptor;
    adaptor.addMeshChannel("grid", domain.mesh, domain.layout).addArray("electric", field);
    adaptor.Initialize();
    field = Vector(1., 2., 3.);
    adaptor.refresh("electric");
    field = Vector(7., 8., 9.);
    adaptor.Execute(1, 0.1);
    const auto root = snapshot_test::captured();
    const char* axes[] = {"x", "y", "z"};
    for (unsigned d = 0; d < 3; ++d) {
        expectConstant(root, std::string("catalyst/channels/grid/data/fields/electric/values/")
                                + axes[d], domain, 1. + d);
    }
    adaptor.Finalize();
}

// Capture after local extents grow and shrink must use the new field layout and
// survive Conduit rebinding without reading stale sizes or overwritten values.
TEST(CatalystFieldSnapshots, ChangedLocalExtents) {
    Domain domain;
    ScalarField field(domain.mesh, domain.layout);
    ippl::CatalystAdaptor adaptor;
    adaptor.addMeshChannel("grid", domain.mesh, domain.layout)
        .addArray("density", field, ippl::UpdatePolicy::Manual);
    adaptor.Initialize();
    field = -1.;
    adaptor.Execute(1, 0.1);
    for (unsigned n : {16u, 4u}) {
        domain.resize(n);
        field.updateLayout(domain.layout);
        field = double(n);
        adaptor.refresh("density");
        field = -double(n);
        adaptor.Execute(n, 0.2);
        expectConstant(snapshot_test::captured(), VALUES, domain, double(n));
    }
    adaptor.Finalize();
}

// With two MPI ranks, uneven partitions of the same global mesh change local
// extents in opposite directions. Each rank must export its newly captured cells.
TEST(CatalystFieldSnapshots, RepartitionOfSameGlobalMesh) {
    if (ippl::Comm->size() != 2) GTEST_SKIP() << "Requires exactly two MPI ranks.";
    Domain domain;
    ScalarField field(domain.mesh, domain.layout);
    ippl::CatalystAdaptor adaptor;
    adaptor.addMeshChannel("grid", domain.mesh, domain.layout)
        .addArray("density", field, ippl::UpdatePolicy::Manual);
    adaptor.Initialize();
    field = -1.;
    adaptor.Execute(1, 0.1);
    for (int split : {6, 2}) {
        std::vector<ippl::NDIndex<3>> partitions{
            {ippl::Index(0, split - 1), ippl::Index(8), ippl::Index(8)},
            {ippl::Index(split, 7), ippl::Index(8), ippl::Index(8)}};
        domain.layout.updateLayout(partitions);
        field.updateLayout(domain.layout);
        const double expected = split + ippl::Comm->rank();
        field = expected;
        adaptor.refresh("density");
        field = -expected;
        adaptor.Execute(split, 0.2);
        expectConstant(snapshot_test::captured(), VALUES, domain, expected);
    }
    adaptor.Finalize();
}

// Changing ghost widths without changing owned extents must capture the current
// interior and preserve x-fastest ordering. Distinct ghost values expose stale offsets.
TEST(CatalystFieldSnapshots, ChangedGhostWidth) {
    Domain domain;
    ScalarField field(domain.mesh, domain.layout);
    ippl::CatalystAdaptor adaptor;
    adaptor.addMeshChannel("grid", domain.mesh, domain.layout)
        .addArray("density", field, ippl::UpdatePolicy::Manual);
    adaptor.Initialize();
    field = -1.;
    adaptor.Execute(1, 0.1);
    for (int g : {2, 1}) {
        field.updateLayout(domain.layout, g);
        auto host = Kokkos::create_mirror_view(field.getView());
        Kokkos::deep_copy(host, -999.);
        const auto local = domain.layout.getLocalNDIndex();
        for (int k = 0; k < local[2].length(); ++k)
            for (int j = 0; j < local[1].length(); ++j)
                for (int i = 0; i < local[0].length(); ++i)
                    host(i + g, j + g, k + g) = i + 10. * j + 100. * k;
        Kokkos::deep_copy(field.getView(), host);
        const auto expected = snapshot_test::interior(field);
        adaptor.refresh("density");
        field = -2.;
        adaptor.Execute(g, 0.2);
        snapshot_test::expectValues(snapshot_test::captured()[VALUES], expected);
    }
    adaptor.Finalize();
}

// A parent particle-channel reset recreates the nested mesh node. Rebinding the
// nested channel must retain its captured values rather than resample the source.
TEST(CatalystFieldSnapshots, NestedBlockRebinding) {
    Domain domain;
    ScalarField field(domain.mesh, domain.layout);
    ippl::MeshChannelT<3> channel("mesh", domain.mesh, domain.layout, 1, false,
                                ippl::GeometryPolicy::Dynamic, "parent");
    channel.addArray("density", field, ippl::UpdatePolicy::Manual);
    Inform info("snapshot test");
    conduit_cpp::Node root;
    field = 2.;
    channel.findArray("density")->refresh(ippl::Comm->rank(), info);
    field = 3.;
    channel.init(root, ippl::Comm->rank());
    channel.execute(root, 1, 0.1, ippl::Comm->rank());
    const std::string path = "catalyst/channels/parent/data/mesh/fields/density/values";
    expectConstant(root, path, domain, 2.);

    root["catalyst/channels/parent"].reset();
    field = 4.;
    channel.findArray("density")->refresh(ippl::Comm->rank(), info);
    field = 5.;
    channel.execute(root, 2, 0.2, ippl::Comm->rank());
    expectConstant(root, path, domain, 4.);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
