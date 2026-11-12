// Unit test: PenningTrap exports deposited density before FFT workspace reuse.
// Runs three real FFT/LeapFrog steps with two MPI ranks by default. An observer
// saves owned source cells immediately after deposition in the default Kokkos
// space, then compares them with the exact capture-backend payload after solve.

constexpr unsigned Dim = 3;
const char* TestName = "CatalystPenningDensity";

#include "Ippl.h"
#include "Manager/datatypes.h"
#include "PenningTrapManager.h"
#include "SnapshotTestUtils.h"
#include <algorithm>
#include <cmath>

namespace {
class ObservedPenning : public PenningTrapManager<double, 3> {
public:
    using Manager = PenningTrapManager<double, 3>;
    using Manager::Manager;
    std::vector<double> density_m;

    void par2grid() override {
        Manager::par2grid();
        density_m = snapshot_test::interior(this->getFieldContainer()->getRho());
    }
};
}  // namespace

// On first and subsequent steps, density must exactly match the deposited field,
// even though FFT GRAD output replaces rho with E_z and diagnostics later reuse it.
TEST(CatalystPenningDensity, DepositedValuesSurviveFFTAndDiagnostics) {
    Vector_t<int, 3> cells(8);
    std::string solver = "FFT";
    std::string step = "LeapFrog";
    ObservedPenning manager(256, 3, cells, 1., solver, step);
    manager.pre_run();
    for (int it = 0; it < 3; ++it) {
        manager.advance();
        const auto root = snapshot_test::captured();
        snapshot_test::expectValues(
            root["catalyst/channels/penning_mesh/data/fields/density/values"], manager.density_m);
        EXPECT_TRUE(std::all_of(manager.density_m.begin(), manager.density_m.end(),
                                [](double value) { return std::isfinite(value); }));
        const auto scalarAfterSolve = snapshot_test::interior(manager.getFieldContainer()->getRho());
        EXPECT_NE(manager.density_m, scalarAfterSolve);
        manager.post_step();
    }
    manager.cat_viz.Finalize();
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    ippl::finalize();
    return result;
}
