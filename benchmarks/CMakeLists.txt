# ------------------------------------------------------------------------------
# benchmarks/CMakeLists.txt
#
# Optional benchmark executables built when IPPL_ENABLE_BENCHMARK is ON.
# ------------------------------------------------------------------------------
message(STATUS "Configuring benchmarks/")

include(AddIpplBenchmark)

add_ippl_benchmark(BenchmarkSmoke SOURCES SmokeBench.cpp REPORT_TIMERS smoke)
