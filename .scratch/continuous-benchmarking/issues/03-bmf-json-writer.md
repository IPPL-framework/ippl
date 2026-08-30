# 03: Implement BMF JSON writer for `IpplTimings`

**What to build:** A small utility that reads the existing `IpplTimings` measurements, filters them against the generated whitelist, and writes a BMF JSON file. The smoke benchmark is updated to emit BMF JSON alongside Google Benchmark JSON.

**Blocked by:** 02

**Status:** resolved

- [ ] A header-only (or small library) BMF writer computes `latency` (mean/min/max) and `count` per timer.
- [ ] Only whitelisted timer names appear in the BMF output.
- [ ] The smoke benchmark produces a valid BMF JSON file.
- [ ] No changes to `IpplTimings` internals.
