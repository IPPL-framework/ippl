#ifndef IPPL_INTERPOLATION_AUTO_TUNE_H
#define IPPL_INTERPOLATION_AUTO_TUNE_H

// ============================================================================
// Width-2 (PIC / CIC) scatter / gather auto-tuning pre-pass.
//
// Default behaviour: NO sweep is run. Sensible per-exec-space defaults are
// seeded into TileSizeCache and GatherCache by Ippl::initialize, so the
// first scatter/gather call uses a known-good configuration without paying
// any benchmarking cost or touching the filesystem.
//
// Opt in to the sweep through the IPPL_AUTO_TUNE env var:
//
//   IPPL_AUTO_TUNE=1 (or "quick")  - small candidate set on a single
//                                    32^3 grid; finishes in ~seconds.
//   IPPL_AUTO_TUNE=full (or "2")   - much broader sweep across grid sizes
//                                    {32, 64, 128}, particle densities
//                                    {0.5, 2, 8, 32} ppc, larger tile /
//                                    team / oversubscription / z_batch
//                                    candidate sets, longer per-config
//                                    measurement. Each density bucket gets
//                                    its own CSV row so the runtime
//                                    density-aware lookup picks the closest
//                                    match. Tens of seconds to a few
//                                    minutes on a GPU.
//
// Anything else (or unset) is treated as no-op. Progress is reported
// through ippl::Info at info level >= 1 (`--info 1`).
//
// On Serial we skip the benchmark entirely (there is nothing to tune -
// Atomic with team_size 1 is the only valid config) and just write a
// single Atomic row so subsequent loads are consistent.
// ============================================================================

#include <string>

namespace ippl::Interpolation::AutoTune {

    /// Populate TileSizeCache and GatherCache with built-in per-exec-space
    /// defaults. Called from ippl::initialize before any scatter/gather is
    /// issued, so the first call has a usable configuration without going
    /// through the auto-tune sweep or reading a CSV from cwd.
    /// No-op if a CSV at one of the standard search paths is already loaded.
    void seedBuiltinDefaults();

    /// Run the width-2 scatter/gather auto-tune pre-pass when explicitly
    /// requested via `IPPL_AUTO_TUNE=1`, write the resulting CSVs next to
    /// the running executable, and return true if a CSV exists at the
    /// requested path on exit. When the env var is not set (default),
    /// returns false without doing any work - callers should fall back to
    /// the seeded defaults in TileSizeCache / GatherCache.
    /// Rank 0 owns the writing; all ranks barrier before returning so any
    /// subsequent cache reload is consistent across ranks.
    bool runOnFirstUse(const std::string& output_path = "tile_sweep_sa_optimal.csv");

    /// One-shot startup hook: seed the built-in defaults and, when
    /// `IPPL_AUTO_TUNE` is set, run the sweep. Called from `ippl::initialize`.
    inline void initialize() {
        seedBuiltinDefaults();
        runOnFirstUse();
    }

}  // namespace ippl::Interpolation::AutoTune

#endif  // IPPL_INTERPOLATION_AUTO_TUNE_H
