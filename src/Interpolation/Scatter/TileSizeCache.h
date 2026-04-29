#ifndef IPPL_TILE_SIZE_CACHE_H
#define IPPL_TILE_SIZE_CACHE_H

// ============================================================================
// TileSizeCache
// ============================================================================
//
// Singleton that loads optimal configurations produced by BenchmarkTileSweep
// and exposes them to Scatter::dispatch at runtime.
//
// Supported CSV files (checked in order):
//   1. $IPPL_TILE_CSV              — explicit path override
//   2. tile_sweep_sa_optimal.csv   — BO-optimised + Atomic results (preferred)
//   3. tile_sweep_optimal.csv      — uniform sweep results (fallback)
//
// ── Formats ──────────────────────────────────────────────────────────────────
//
// Uniform sweep (write_optimal_csv), with rho column:
//   method,kernel_width,rho,optimal_tile_size,throughput_Mpts_s,time_ms
//
// BO/Atomic (write_bo_csv), with rho column:
//   method,value_type,kernel_width,rho,
//   best_tile_x,best_tile_y,best_tile_z,
//   best_team_size,best_oversubscription_factor,best_z_batches,
//   throughput_Mpts_s,time_ms,kernel_evaluations,preflight_rejections
//
// Both formats are also parseable WITHOUT the rho column (old files).
// In that case rho defaults to 0.0 ("unspecified").
//
// ── Density-aware lookup ─────────────────────────────────────────────────────
//
// Multiple rows with the same (method, width, is_complex) but different rho
// values coexist.  get() / get_best() accept an optional rho parameter:
//   rho <= 0  → return the entry with the highest throughput (ignore density)
//   rho >  0  → return the entry whose recorded rho is closest to the query
//               (entries with rho=0 are a low-priority fallback)
//
// ── Method selection ─────────────────────────────────────────────────────────
//
// get_best(width, is_complex, rho) returns the method+config with the highest
// throughput for the query density across Atomic, Tiled, and OutputFocused.
//
// Thread safety: initialised once via std::call_once, then read-only.
// ============================================================================

// Header keeps only the type/API surface and the inline lookup logic.
// CSV parsing, file IO, and load() live in TileSizeCache.cpp so that every
// TU including this header doesn't drag in <fstream>/<sstream>.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Interpolation/Scatter/ScatterConfig.h"

namespace ippl {
namespace Interpolation {

// ── Cache key ────────────────────────────────────────────────────────────────
struct TileCacheKey {
    ScatterMethod method;
    int           kernel_width;
    bool          is_complex;

    bool operator==(const TileCacheKey& o) const noexcept {
        return method == o.method && kernel_width == o.kernel_width
               && is_complex == o.is_complex;
    }
};

struct TileCacheKeyHash {
    std::size_t operator()(const TileCacheKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.method);
        h             = h * 31 + static_cast<std::size_t>(k.kernel_width);
        h             = h * 31 + static_cast<std::size_t>(k.is_complex);
        return h;
    }
};

// ── Cache entry ───────────────────────────────────────────────────────────────
struct TileCacheEntry {
    std::array<int, 3> tile     = {1, 1, 1};
    int  team_size              = -1;   // -1 → keep ScatterConfig default
    int  oversubscription_factor = -1;  // -1 → keep ScatterConfig default
    int  z_batches              = -1;   // -1 → keep ScatterConfig default
    bool is_rectangular         = false;
    double throughput_Mpts_s    = 0.0;  // used for best-method selection
    double rho                  = 0.0;  // 0 = unspecified (old CSV, no rho col)
};

// ── Result of get_best() ─────────────────────────────────────────────────────
struct BestCacheEntry {
    ScatterMethod  method;
    TileCacheEntry entry;
};

// ─────────────────────────────────────────────────────────────────────────────
class TileSizeCache {
public:
    static TileSizeCache& instance() {
        static TileSizeCache inst;
        std::call_once(inst.init_flag_, [&]() { inst.load(); });
        return inst;
    }

    // ------------------------------------------------------------------
    // Lookup by explicit method + optional density hint.
    // ------------------------------------------------------------------
    std::optional<TileCacheEntry> get(ScatterMethod method, int kernel_width,
                                      bool is_complex, double rho = 0.0) const {
        TileCacheKey key{method, kernel_width, is_complex};
        auto it = entries_.find(key);
        if (it == entries_.end() || it->second.empty())
            return std::nullopt;
        const TileCacheEntry* e = closest_entry(it->second, rho);
        return e ? std::optional<TileCacheEntry>(*e) : std::nullopt;
    }

    // ------------------------------------------------------------------
    // Auto method selection: best throughput across ALL methods at the
    // closest recorded density.
    // ------------------------------------------------------------------
    std::optional<BestCacheEntry> get_best(int kernel_width, bool is_complex,
                                           double rho = 0.0) const {
        std::optional<BestCacheEntry> best;
        double best_tp = -1.0;

        for (const auto& [key, vec] : entries_) {
            if (key.kernel_width != kernel_width || key.is_complex != is_complex)
                continue;
            if (vec.empty())
                continue;
            const TileCacheEntry* e = closest_entry(vec, rho);
            if (e && e->throughput_Mpts_s > best_tp) {
                best_tp = e->throughput_Mpts_s;
                best    = BestCacheEntry{key.method, *e};
            }
        }
        return best;
    }

    // Convenience: tile-only lookup (backward compat).
    template <unsigned Dim>
    std::optional<Vector<int, Dim>> get_tile(ScatterMethod method, int kernel_width,
                                             bool is_complex, double rho = 0.0) const {
        auto entry = get(method, kernel_width, is_complex, rho);
        if (!entry.has_value())
            return std::nullopt;
        const auto& e = entry.value();
        Vector<int, Dim> tile;
        for (unsigned d = 0; d < Dim; ++d)
            tile[d] = e.tile[d < 3 ? d : 2];
        return tile;
    }

    bool loaded() const noexcept { return loaded_; }
    const std::string& source() const noexcept { return source_path_; }

    /// Drop all cached entries.  Useful for benchmarks (e.g. TileSweep) that
    /// must run scatter with explicit configs, not cached ones.
    void clear() {
        entries_.clear();
        loaded_      = false;
        source_path_ = "";
    }

    void reload(const std::string& path = "");

    /// Seed a single (method, kernel_width, is_complex) entry. Idempotent
    /// per row: an existing entry with the same density bucket and a higher
    /// throughput value wins. Used by Ippl::initialize to pre-populate the
    /// cache with built-in per-exec-space defaults so the first scatter
    /// does not pay the AutoTune cost.
    void seed_default(ScatterMethod method, int kernel_width, bool is_complex,
                      TileCacheEntry entry) {
        insert_entry(method, kernel_width, is_complex, std::move(entry));
        loaded_ = true;
        if (source_path_.empty()) {
            source_path_ = "<built-in default>";
        }
    }

private:
    TileSizeCache() = default;

    std::once_flag init_flag_;
    // Key = (method, width, is_complex); value = all density variants.
    std::unordered_map<TileCacheKey, std::vector<TileCacheEntry>, TileCacheKeyHash> entries_;
    bool        loaded_      = false;
    std::string source_path_ = "";

    // ------------------------------------------------------------------
    // Density-aware entry selection.
    //   rho <= 0 → highest throughput among all entries
    //   rho >  0 → entry with closest positive rho; rho=0 entries are fallback
    // ------------------------------------------------------------------
    static const TileCacheEntry* closest_entry(const std::vector<TileCacheEntry>& vec,
                                                double rho) {
        if (vec.empty())
            return nullptr;

        if (rho <= 0.0) {
            return &*std::max_element(vec.begin(), vec.end(),
                [](const TileCacheEntry& a, const TileCacheEntry& b) {
                    return a.throughput_Mpts_s < b.throughput_Mpts_s;
                });
        }

        const TileCacheEntry* best      = nullptr;
        double                best_dist = std::numeric_limits<double>::max();

        for (const auto& e : vec) {
            // rho=0 entries used only when no positive-rho entry exists
            double dist = (e.rho > 0.0) ? std::abs(e.rho - rho) : 1e30;
            if (dist < best_dist) {
                best_dist = dist;
                best      = &e;
            }
        }
        // If only rho=0 entries exist, fall back to highest-throughput among them
        if (best_dist >= 1e29) {
            return &*std::max_element(vec.begin(), vec.end(),
                [](const TileCacheEntry& a, const TileCacheEntry& b) {
                    return a.throughput_Mpts_s < b.throughput_Mpts_s;
                });
        }
        return best;
    }

    // ------------------------------------------------------------------
    // Insert with conflict resolution.
    // Rows whose rho values are within 0.1% are treated as the same density
    // bucket and the higher-throughput entry wins.
    // ------------------------------------------------------------------
    void insert_entry(ScatterMethod method, int width, bool is_complex,
                      TileCacheEntry entry);

    // ------------------------------------------------------------------
    // File discovery and CSV parsing — defined in TileSizeCache.cpp.
    // ------------------------------------------------------------------
    void load();
    bool load_file(const std::string& path);
    bool parse_uniform_row(const std::string& line, bool has_rho);
    bool parse_rect_row(const std::string& line, bool has_rho);
};

}  // namespace Interpolation
}  // namespace ippl

#endif  // IPPL_TILE_SIZE_CACHE_H
