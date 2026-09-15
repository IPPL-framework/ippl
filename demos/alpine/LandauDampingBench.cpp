// ------------------------------------------------------------------------------
// LandauDampingBench
//
// Google Benchmark executable for the LandauDampingManager. It reuses the same
// physics setup as the LandauDamping demo and measures the cost of one
// manager->advance() call per benchmark iteration.
// ------------------------------------------------------------------------------

constexpr unsigned Dim = 3;
using T                = double;
const char* TestName   = "LandauDampingBench";

#include "Ippl.h"

#include <benchmark/benchmark.h>
#include <memory>
#include <string>
#include <vector>

#include "Manager/datatypes.h"

#include "Utility/BenchmarkMetrics.h"
#include "Utility/IpplTimings.h"

#include "LandauDampingBench_timers.h"
#include "LandauDampingManager.h"
#include "Manager/PicManager.h"

struct BenchConfig {
    Vector_t<int, Dim> nr;
    size_type totalP;
    int nt;
    std::string solver;
    double lbt;
    std::string stepMethod;
    std::vector<std::string> preconditionerParams;
};

static BenchConfig config_m;

class LandauDampingFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State&) override {
        manager_m = std::make_unique<LandauDampingManager<T, Dim>>(
            config_m.totalP, config_m.nt, config_m.nr, config_m.lbt, config_m.solver,
            config_m.stepMethod, config_m.preconditionerParams);

        manager_m->pre_run();
        IpplTimings::resetAllTimers();
    }

    void TearDown(const ::benchmark::State&) override {
        IpplTimings::print();
        ippl::benchmark::writeBMF("LandauDampingBench.bmf.json", reportedTimers);

        // Destroy the manager before ippl::finalize() so Kokkos views are
        // deallocated while Kokkos is still initialized.
        manager_m.reset();
    }

    std::unique_ptr<LandauDampingManager<T, Dim>> manager_m;
};

BENCHMARK_DEFINE_F(LandauDampingFixture, BM_Advance)(benchmark::State& state) {
    for (auto _ : state) {
        manager_m->advance();
    }
}

BENCHMARK_REGISTER_F(LandauDampingFixture, BM_Advance);

int main(int argc, char** argv) {
    ippl::initialize(argc, argv);

    Inform msg(TestName);

    int arg = 1;
    for (unsigned d = 0; d < Dim; d++) {
        config_m.nr[d] = std::atoi(argv[arg++]);
    }

    config_m.totalP     = std::atoll(argv[arg++]);
    config_m.nt         = std::atoi(argv[arg++]);
    config_m.solver     = argv[arg++];
    config_m.lbt        = std::atof(argv[arg++]);
    config_m.stepMethod = argv[arg++];

    if (config_m.solver == "PCG" || config_m.solver == "FEM_PRECON") {
        while (arg < argc) {
            const std::string token = argv[arg];
            if (token.rfind("--", 0) == 0) {
                break;
            }
            config_m.preconditionerParams.push_back(token);
            ++arg;
        }
    }

    // Consume demo-specific trailing flags (--overallocate, --info) and pass
    // the rest to Google Benchmark.
    std::vector<char*> benchArgv;
    benchArgv.push_back(argv[0]);
    for (int i = arg; i < argc;) {
        const std::string token = argv[i];
        if (token == "--overallocate" || token == "--info") {
            i += 2;
        } else {
            benchArgv.push_back(argv[i]);
            ++i;
        }
    }
    int benchArgc = static_cast<int>(benchArgv.size());

    ::benchmark::Initialize(&benchArgc, benchArgv.data());
    if (::benchmark::ReportUnrecognizedArguments(benchArgc, benchArgv.data())) {
        ippl::finalize();
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    ippl::finalize();
    return 0;
}
