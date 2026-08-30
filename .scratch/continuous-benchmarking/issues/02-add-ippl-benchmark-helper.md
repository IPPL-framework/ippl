# 02: Implement `add_ippl_benchmark()` CMake helper

**What to build:** A reusable CMake function that creates a benchmark target, accepts `REPORT_TIMERS`/`IGNORE_TIMERS`, and generates a compile-time timer-whitelist header. The smoke benchmark from ticket 01 is converted to use this helper.

**Blocked by:** 01

**Status:** resolved

- [ ] `cmake/AddIpplBenchmark.cmake` defines `add_ippl_benchmark(<name> SOURCES ... [REPORT_TIMERS ...] [IGNORE_TIMERS ...])`.
- [ ] The function generates `<name>_timers.h` in the build tree containing a `constexpr std::array<std::string_view, N> reportedTimers`.
- [ ] Supplying both `REPORT_TIMERS` and `IGNORE_TIMERS` is a CMake configure error.
- [ ] The smoke benchmark uses the helper and compiles with a whitelist.
