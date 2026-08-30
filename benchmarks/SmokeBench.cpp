// ------------------------------------------------------------------------------
// Smoke benchmark for IPPL's Google Benchmark integration.
//
// This is a minimal executable used to verify that the CMake FetchContent setup,
// compile flags, benchmark::benchmark linking, add_ippl_benchmark() timer
// whitelist generation, and BenchmarkMetrics BMF writing all work correctly.
// ------------------------------------------------------------------------------

#include "Ippl.h"

#include <benchmark/benchmark.h>

#include "Utility/BenchmarkMetrics.h"
#include "Utility/IpplTimings.h"

#include "BenchmarkSmoke_timers.h"

class SmokeFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State&) override { IpplTimings::resetAllTimers(); }

    void TearDown(const ::benchmark::State&) override {
        ippl::benchmark::writeBMF("BenchmarkSmoke.bmf.json", reportedTimers);
    }
};

BENCHMARK_DEFINE_F(SmokeFixture, BM_Smoke)(benchmark::State& state) {
    static IpplTimings::TimerRef smokeTimer = IpplTimings::getTimer("smoke");

    for (auto _ : state) {
        IpplTimings::startTimer(smokeTimer);

        int sum = 0;
        for (int i = 0; i < 100; ++i) {
            sum += i;
        }
        benchmark::DoNotOptimize(sum);

        IpplTimings::stopTimer(smokeTimer);
    }
}

BENCHMARK_REGISTER_F(SmokeFixture, BM_Smoke);

int main(int argc, char** argv) {
    ippl::initialize(argc, argv);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        ippl::finalize();
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    ippl::finalize();
    return 0;
}
