/*!
 * @file TileSizeCache.cpp
 * @brief CSV parsing and file IO for the singleton tile-size cache.
 *
 * Header-only API lives in TileSizeCache.h; this TU keeps everything that
 * pulls in @c <fstream> / @c <sstream>.
 */

#include "Interpolation/Scatter/TileSizeCache.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Interpolation/Gather/GatherConfig.h"
#include "Ippl.h"

#if __has_include("IpplAutoTunePresets.h")
#include "IpplAutoTunePresets.h"
#endif

namespace ippl::Interpolation {

namespace {

    bool parse_method(const std::string& s, ScatterMethod& out) {
        if (s == "Tiled")         { out = ScatterMethod::Tiled;         return true; }
        if (s == "OutputFocused") { out = ScatterMethod::OutputFocused; return true; }
        if (s == "Atomic")        { out = ScatterMethod::Atomic;        return true; }
        return false;
    }

    int parse_int(const std::string& s) {
        try { return std::stoi(s); } catch (...) { return -1; }
    }

    double parse_double(const std::string& s) {
        try { return std::stod(s); } catch (...) { return 0.0; }
    }

    std::vector<std::string> split_csv(const std::string& line) {
        std::vector<std::string> out;
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            auto l = tok.find_first_not_of(" \t\r\n");
            auto r = tok.find_last_not_of(" \t\r\n");
            out.push_back(l == std::string::npos ? "" : tok.substr(l, r - l + 1));
        }
        return out;
    }

}  // namespace

void TileSizeCache::reload(const std::string& path) {
    entries_.clear();
    loaded_      = false;
    source_path_ = "";
    if (!path.empty())
        load_file(path);
    else
        load();
}

void TileSizeCache::insert_entry(ScatterMethod method, int width, bool is_complex,
                                 TileCacheEntry entry) {
    TileCacheKey key{method, width, is_complex};
    auto&        vec = entries_[key];

    constexpr double rho_tol_frac = 0.001;
    for (auto& existing : vec) {
        double tol = rho_tol_frac * std::max({existing.rho, entry.rho, 1.0});
        if (std::abs(existing.rho - entry.rho) <= tol) {
            // Same density bucket: rectangular beats uniform; higher tp wins
            if (!existing.is_rectangular && entry.is_rectangular)
                existing = entry;
            else if (existing.is_rectangular == entry.is_rectangular
                     && entry.throughput_Mpts_s > existing.throughput_Mpts_s)
                existing = entry;
            return;
        }
    }
    vec.push_back(std::move(entry));
}

void TileSizeCache::load() {
    if (const char* env = std::getenv("IPPL_TILE_CSV")) {
        if (load_file(std::string(env)))
            return;
        if (ippl::Warn) {
            *ippl::Warn << "[TileSizeCache] IPPL_TILE_CSV=" << env
                        << " could not be read, falling back." << endl;
        }
    }
    if (load_file("tile_sweep_sa_optimal.csv"))
        return;
    if (load_file("tile_sweep_optimal.csv"))
        return;

#ifdef IPPL_AUTOTUNE_PRESET_DIR
    // Shipped preset baked in at configure time for this build's arch.
    {
        const std::string preset =
            std::string(IPPL_AUTOTUNE_PRESET_DIR) + "/tile_sweep_sa_optimal.csv";
        if (load_file(preset)) {
            if (ippl::Info) {
                *ippl::Info << ::level2
                            << "[TileSizeCache] using shipped preset for "
                            << IPPL_AUTOTUNE_ARCH_TAG << " (" << preset << ")" << endl;
            }
            return;
        }
    }
#endif
}

bool TileSizeCache::load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        return false;

    std::string line;
    if (!std::getline(f, line))
        return false;

    const bool is_rect    = (line.find("best_tile_x")     != std::string::npos);
    const bool is_uniform = (line.find("optimal_tile_size") != std::string::npos);
    if (!is_rect && !is_uniform)
        return false;

    const bool has_rho = (line.find(",rho,") != std::string::npos);

    int rows = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        rows += is_rect ? (parse_rect_row(line, has_rho)    ? 1 : 0)
                        : (parse_uniform_row(line, has_rho)  ? 1 : 0);
    }
    if (rows == 0)
        return false;

    loaded_      = true;
    source_path_ = path;
    return true;
}

// ------------------------------------------------------------------
// Uniform-format row:
//   with rho:    method,kernel_width,rho,optimal_tile,throughput,time
//   without rho: method,kernel_width,optimal_tile,throughput,time
// ------------------------------------------------------------------
bool TileSizeCache::parse_uniform_row(const std::string& line, bool has_rho) {
    auto fields = split_csv(line);
    // Need at least: method, width, [rho,] tile
    const int min_f = has_rho ? 4 : 3;
    if ((int)fields.size() < min_f)
        return false;

    ScatterMethod method;
    if (!parse_method(fields[0], method))
        return false;

    int    width    = parse_int(fields[1]);
    double rho      = has_rho ? parse_double(fields[2]) : 0.0;
    int    tile_col = has_rho ? 3 : 2;
    int    tp_col   = tile_col + 1;

    int    tile       = parse_int(fields[tile_col]);
    double throughput = ((int)fields.size() > tp_col) ? parse_double(fields[tp_col]) : 0.0;

    if (width <= 0 || tile <= 0)
        return false;

    // Uniform has no value_type column -> insert for both
    for (bool cx : {false, true}) {
        TileCacheEntry e;
        e.tile.fill(tile);
        e.is_rectangular          = false;
        e.team_size               = -1;
        e.oversubscription_factor = -1;
        e.z_batches               = -1;
        e.throughput_Mpts_s       = throughput;
        e.rho                     = rho;
        insert_entry(method, width, cx, e);
    }
    return true;
}

// ------------------------------------------------------------------
// BO/Atomic-format row:
//   with rho:
//     method,value_type,kernel_width,rho,
//     tx,ty,tz, team_size,osub,z_batches,
//     throughput,time,kernel_evals,preflight
//     col: 0  1          2            3
//          4  5  6       7     8    9
//          10  11  12   13
//   without rho: same with cols from 3 shifted left by 1
// ------------------------------------------------------------------
bool TileSizeCache::parse_rect_row(const std::string& line, bool has_rho) {
    auto fields = split_csv(line);
    // Minimum: through z_batches (col 9 with rho, col 8 without)
    const int min_f = has_rho ? 10 : 9;
    if ((int)fields.size() < min_f)
        return false;

    ScatterMethod method;
    if (!parse_method(fields[0], method))
        return false;

    const std::string& vtype = fields[1];
    int    width    = parse_int(fields[2]);
    double rho      = has_rho ? parse_double(fields[3]) : 0.0;
    int    tc       = has_rho ? 4 : 3;  // tile column start

    if ((int)fields.size() < tc + 6)
        return false;

    int tx     = parse_int(fields[tc]);
    int ty     = parse_int(fields[tc + 1]);
    int tz     = parse_int(fields[tc + 2]);
    int team   = parse_int(fields[tc + 3]);
    int osub   = parse_int(fields[tc + 4]);
    int zb     = parse_int(fields[tc + 5]);
    int tp_col = tc + 6;

    double throughput = ((int)fields.size() > tp_col) ? parse_double(fields[tp_col]) : 0.0;

    if (width <= 0 || tx <= 0 || ty <= 0 || tz <= 0)
        return false;

    bool is_cx = (vtype == "complex");

    auto do_insert = [&](bool cx) {
        TileCacheEntry e;
        e.tile                    = {tx, ty, tz};
        e.is_rectangular          = (tx != ty || ty != tz);
        e.team_size               = (team > 0) ? team : -1;
        e.oversubscription_factor = (osub > 0)  ? osub : -1;
        e.z_batches               = (zb   > 0)  ? zb   : -1;
        e.throughput_Mpts_s       = throughput;
        e.rho                     = rho;
        insert_entry(method, width, cx, e);
    };

    if (vtype.empty()) {
        do_insert(false);
        do_insert(true);
    } else {
        do_insert(is_cx);
    }
    return true;
}

void GatherCache::load() {
    auto try_load = [&](const std::string& path) -> bool {
        std::ifstream f(path);
        if (!f.is_open()) return false;

        std::string line;
        if (!std::getline(f, line)) return false;  // header
        if (!std::getline(f, line)) return false;  // single data row

        auto fields = split_csv(line);
        if (fields.size() < 5) return false;

        if (fields[0] == "AtomicSort") {
            entry_.method = GatherMethod::AtomicSort;
        } else {
            entry_.method = GatherMethod::Atomic;
        }

        entry_.tile = {parse_int(fields[2]), parse_int(fields[3]), parse_int(fields[4])};
        for (auto& v : entry_.tile) {
            if (v <= 0) v = 1;
        }
        loaded_ = true;
        return true;
    };

    if (const char* env = std::getenv("IPPL_GATHER_CSV")) {
        if (try_load(std::string(env))) return;
    }
    if (try_load("gather_sweep_optimal.csv")) return;

#ifdef IPPL_AUTOTUNE_PRESET_DIR
    const std::string preset =
        std::string(IPPL_AUTOTUNE_PRESET_DIR) + "/gather_sweep_optimal.csv";
    if (try_load(preset) && ippl::Info) {
        *ippl::Info << ::level2
                    << "[GatherCache] using shipped preset for "
                    << IPPL_AUTOTUNE_ARCH_TAG << " (" << preset << ")" << endl;
    }
#endif
}

}  // namespace ippl::Interpolation
