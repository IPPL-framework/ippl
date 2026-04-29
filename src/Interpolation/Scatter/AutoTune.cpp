// ============================================================================
// Width-2 scatter auto-tuner — see AutoTune.h for the user-visible contract.
// ============================================================================

#include "Ippl.h"

#include "Interpolation/Scatter/AutoTune.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Field/Field.h"
#include "FieldLayout/FieldLayout.h"
#include "Interpolation/Gather/GatherConfig.h"
#include "Interpolation/Kernels.h"
#include "Interpolation/Scatter/ScatterConfig.h"
#include "Meshes/UniformCartesian.h"
#include "Particle/ParticleAttrib.h"
#include "Particle/ParticleBase.h"
#include "Particle/ParticleSpatialLayout.h"

#include "Interpolation/Gather/Gather.h"
#include "Interpolation/Gather/GatherConfig.h"
#include "Interpolation/Scatter/Scatter.h"
#include "Interpolation/Scatter/TileSizeCache.h"

namespace ippl::Interpolation::AutoTune {

    namespace {

        struct Sample {
            ScatterMethod method;
            std::string   value_type;  // "real" | "complex"
            int           kernel_width;
            double        rho;
            int           tile_x, tile_y, tile_z;
            int           team_size;
            int           osub;
            int           z_batches;
            double        throughput_Mpts_s;
            double        time_ms;
        };

        struct GatherSample {
            GatherMethod method;
            int          tile_x, tile_y, tile_z;
            double       throughput_Mpts_s;
        };

        const char* gather_method_name(GatherMethod m) {
            switch (m) {
                case GatherMethod::Atomic:     return "Atomic";
                case GatherMethod::AtomicSort: return "AtomicSort";
            }
            return "Atomic";
        }

        void write_gather_csv(const std::string& path, const GatherSample& s) {
            std::ofstream out(path);
            if (!out.is_open()) return;
            out << "method,kernel_width,tile_x,tile_y,tile_z,throughput_Mpts_s\n"
                << std::fixed << std::setprecision(2)
                << gather_method_name(s.method) << ",2,"
                << s.tile_x << "," << s.tile_y << "," << s.tile_z << ","
                << s.throughput_Mpts_s << "\n";
        }

        const char* method_name(ScatterMethod m) {
            switch (m) {
                case ScatterMethod::Atomic:        return "Atomic";
                case ScatterMethod::Tiled:         return "Tiled";
                case ScatterMethod::OutputFocused: return "OutputFocused";
            }
            return "Atomic";
        }

        void write_csv(const std::string& path, const std::vector<Sample>& samples) {
            const std::string header =
                "method,value_type,kernel_width,rho,best_tile_x,best_tile_y,best_tile_z,"
                "best_team_size,best_oversubscription_factor,best_z_batches,"
                "throughput_Mpts_s,time_ms,kernel_evaluations,preflight_rejections";
            std::ofstream out(path);
            if (!out.is_open()) return;
            out << header << "\n" << std::fixed;
            for (const auto& s : samples) {
                out << method_name(s.method) << "," << s.value_type << "," << s.kernel_width
                    << "," << std::setprecision(4) << s.rho << ","
                    << s.tile_x << "," << s.tile_y << "," << s.tile_z << ","
                    << s.team_size << "," << s.osub << "," << s.z_batches << ","
                    << std::setprecision(2) << s.throughput_Mpts_s << ","
                    << std::setprecision(4) << s.time_ms << ",0,0\n";
            }
        }

        // Trivial CSV for backends with nothing to tune: Atomic only, team=1.
        void write_trivial(const std::string& path) {
            std::vector<Sample> samples;
            for (int w : {1, 2}) {
                for (const char* vt : {"real", "complex"}) {
                    samples.push_back(
                        {ScatterMethod::Atomic, vt, w, 0.0, 1, 1, 1, 1, 1, 1, 0.0, 0.0});
                }
            }
            write_csv(path, samples);
        }

        void write_trivial_gather(const std::string& path) {
            write_gather_csv(path, GatherSample{GatherMethod::Atomic, 1, 1, 1, 0.0});
        }

        template <typename ExecSpace>
        double time_config(ScatterMethod method, const Vector<int, 3>& tile_size,
                           int team_size, int osub, int z_batches,
                           unsigned N = 32, size_t nParticle = 100000, int runs = 3) {
            using value_t   = double;
            using mesh_t    = ippl::UniformCartesian<value_t, 3>;
            using center_t  = typename mesh_t::DefaultCentering;
            using field_t   = ippl::Field<value_t, 3, mesh_t, center_t, ExecSpace>;
            using flayout_t = ippl::FieldLayout<3>;
            using playout_t = ippl::ParticleSpatialLayout<value_t, 3, mesh_t, ExecSpace>;

            ippl::Index Ix(N), Iy(N), Iz(N);
            ippl::NDIndex<3> dom(Ix, Iy, Iz);
            std::array<bool, 3> isParallel = {true, true, true};
            ippl::Vector<value_t, 3> hx{1.0 / value_t(N), 1.0 / value_t(N), 1.0 / value_t(N)};
            ippl::Vector<value_t, 3> origin{0, 0, 0};

            flayout_t layout(MPI_COMM_WORLD, dom, isParallel);
            mesh_t    mesh(dom, hx, origin);
            field_t   field(mesh, layout);
            field = 0.0;

            playout_t playout(layout, mesh);

            struct Bunch : ippl::ParticleBase<playout_t> {
                explicit Bunch(playout_t& pl) : ippl::ParticleBase<playout_t>(pl) {
                    this->addAttribute(Q);
                }
                ippl::ParticleAttrib<value_t, ExecSpace> Q;
            };
            Bunch bunch(playout);
            const size_t nLoc = nParticle / std::max(1, ippl::Comm->size());
            bunch.create(nLoc);

            auto R_host = bunch.R.getHostMirror();
            std::mt19937_64 eng(42 + ippl::Comm->rank());
            std::uniform_real_distribution<value_t> u(0.01, 0.99);
            for (size_t i = 0; i < nLoc; ++i) {
                R_host(i)[0] = u(eng);
                R_host(i)[1] = u(eng);
                R_host(i)[2] = u(eng);
            }
            Kokkos::deep_copy(bunch.R.getView(), R_host);
            bunch.Q = 1.0;
            bunch.update();

            ScatterConfig<3> cfg = ScatterConfig<3>::template get_default<ExecSpace>();
            cfg.method           = method;
            cfg.set_tile_size(tile_size);
            if (team_size > 0) cfg.team_size = team_size;
            if (osub > 0)      cfg.oversubscription_factor = osub;
            if (z_batches > 0) cfg.z_batches = z_batches;
            cfg.lock_method   = true;
            cfg.enable_tuning = false;

            ippl::Interpolation::LinearKernel<value_t> cic;

            field = 0.0;
            bunch.Q.scatter_kernel(field, bunch.R, cic, cfg);
            Kokkos::fence();

            double best_ms = 1e18;
            for (int r = 0; r < runs; ++r) {
                field = 0.0;
                Kokkos::fence();
                auto t0 = std::chrono::steady_clock::now();
                bunch.Q.scatter_kernel(field, bunch.R, cic, cfg);
                Kokkos::fence();
                auto t1 = std::chrono::steady_clock::now();
                const double ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best_ms) best_ms = ms;
            }
            return (best_ms > 0) ? double(nLoc) / 1e6 / (best_ms * 1e-3) : 0.0;
        }

        template <typename ExecSpace>
        double time_gather(GatherMethod method, const Vector<int, 3>& tile_size,
                           unsigned N = 32, size_t nParticle = 100000, int runs = 3) {
            using value_t   = double;
            using mesh_t    = ippl::UniformCartesian<value_t, 3>;
            using center_t  = typename mesh_t::DefaultCentering;
            using field_t   = ippl::Field<value_t, 3, mesh_t, center_t, ExecSpace>;
            using flayout_t = ippl::FieldLayout<3>;
            using playout_t = ippl::ParticleSpatialLayout<value_t, 3, mesh_t, ExecSpace>;

            ippl::Index Ix(N), Iy(N), Iz(N);
            ippl::NDIndex<3> dom(Ix, Iy, Iz);
            std::array<bool, 3> isParallel = {true, true, true};
            ippl::Vector<value_t, 3> hx{1.0 / value_t(N), 1.0 / value_t(N), 1.0 / value_t(N)};
            ippl::Vector<value_t, 3> origin{0, 0, 0};

            flayout_t layout(MPI_COMM_WORLD, dom, isParallel);
            mesh_t    mesh(dom, hx, origin);
            field_t   field(mesh, layout);
            field = 1.0;

            playout_t playout(layout, mesh);
            struct Bunch : ippl::ParticleBase<playout_t> {
                explicit Bunch(playout_t& pl) : ippl::ParticleBase<playout_t>(pl) {
                    this->addAttribute(Q);
                }
                ippl::ParticleAttrib<value_t, ExecSpace> Q;
            };
            Bunch bunch(playout);
            const size_t nLoc = nParticle / std::max(1, ippl::Comm->size());
            bunch.create(nLoc);

            auto R_host = bunch.R.getHostMirror();
            std::mt19937_64 eng(42 + ippl::Comm->rank());
            std::uniform_real_distribution<value_t> u(0.01, 0.99);
            for (size_t i = 0; i < nLoc; ++i) {
                R_host(i)[0] = u(eng);
                R_host(i)[1] = u(eng);
                R_host(i)[2] = u(eng);
            }
            Kokkos::deep_copy(bunch.R.getView(), R_host);
            bunch.Q = 0.0;
            bunch.update();

            GatherConfig<3> cfg = GatherConfig<3>::template get_default<ExecSpace>();
            cfg.method = method;
            cfg.set_tile_size({tile_size[0], tile_size[1], tile_size[2]});

            ippl::Interpolation::LinearKernel<value_t> cic;
            ippl::Gather<decltype(cic), 3> gather_op(cic, cfg);

            gather_op(field, bunch.R, bunch.Q);
            Kokkos::fence();

            double best_ms = 1e18;
            for (int r = 0; r < runs; ++r) {
                Kokkos::fence();
                auto t0 = std::chrono::steady_clock::now();
                gather_op(field, bunch.R, bunch.Q);
                Kokkos::fence();
                auto t1 = std::chrono::steady_clock::now();
                const double ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best_ms) best_ms = ms;
            }
            return (best_ms > 0) ? double(nLoc) / 1e6 / (best_ms * 1e-3) : 0.0;
        }

        template <typename ExecSpace>
        GatherSample sweep_gather() {
            GatherSample best{GatherMethod::Atomic, 1, 1, 1, 0.0};

            const double atomic_tp = time_gather<ExecSpace>(GatherMethod::Atomic, {1, 1, 1});
            if (atomic_tp > best.throughput_Mpts_s) {
                best = GatherSample{GatherMethod::Atomic, 1, 1, 1, atomic_tp};
            }

            for (const auto& t : std::vector<Vector<int, 3>>{{4, 4, 4}, {8, 8, 8}}) {
                const double tp = time_gather<ExecSpace>(GatherMethod::AtomicSort, t);
                if (tp > best.throughput_Mpts_s) {
                    best = GatherSample{GatherMethod::AtomicSort, t[0], t[1], t[2], tp};
                }
            }
            return best;
        }

        template <typename ExecSpace>
        std::vector<Sample> sweep() {
            std::vector<Sample> out;

            const bool host_backend =
#ifdef KOKKOS_ENABLE_OPENMP
                std::is_same_v<ExecSpace, Kokkos::OpenMP>
#else
                false
#endif
                ;
            const int default_team = host_backend ? 1 : 32;

            // Atomic — tile is irrelevant to the dispatcher; fix to (1,1,1).
            {
                Vector<int, 3> tile{1, 1, 1};
                const double tp =
                    time_config<ExecSpace>(ScatterMethod::Atomic, tile, default_team, 1, 1);
                out.push_back(Sample{ScatterMethod::Atomic, "real", 2, 0.0,
                                     1, 1, 1, default_team, 1, 1, tp, 0.0});
            }

            // Tiled — small candidate set over (tile, team, osub).
            {
                Sample best{ScatterMethod::Tiled, "real", 2, 0.0, 4, 4, 4,
                            host_backend ? 1 : 64, 1, 1, 0.0, 0.0};
                for (const auto& t :
                     std::vector<Vector<int, 3>>{{2, 2, 2}, {4, 4, 4}, {8, 8, 8}}) {
                    const double tp = time_config<ExecSpace>(ScatterMethod::Tiled, t,
                                                             host_backend ? 1 : 64, 1, 1);
                    if (tp > best.throughput_Mpts_s) {
                        best = Sample{ScatterMethod::Tiled, "real", 2, 0.0, t[0], t[1], t[2],
                                      host_backend ? 1 : 64, 1, 1, tp, 0.0};
                    }
                }
                out.push_back(best);
            }

            // OutputFocused — tile + z_batches.
            {
                Sample best{ScatterMethod::OutputFocused, "real", 2, 0.0, 2, 2, 2,
                            host_backend ? 1 : 128, 1, 1, 0.0, 0.0};
                for (const auto& t : std::vector<Vector<int, 3>>{{2, 2, 2}, {4, 4, 4}}) {
                    for (int zb : {1, 4}) {
                        const double tp = time_config<ExecSpace>(
                            ScatterMethod::OutputFocused, t, host_backend ? 1 : 128, 1, zb);
                        if (tp > best.throughput_Mpts_s) {
                            best = Sample{ScatterMethod::OutputFocused, "real", 2, 0.0,
                                          t[0], t[1], t[2],
                                          host_backend ? 1 : 128, 1, zb, tp, 0.0};
                        }
                    }
                }
                out.push_back(best);
            }

            // Mirror real → complex.
            const size_t base = out.size();
            for (size_t i = 0; i < base; ++i) {
                Sample s     = out[i];
                s.value_type = "complex";
                out.push_back(s);
            }
            return out;
        }

        // ====================================================================
        // FULL sweep — broader candidate set across grid sizes, particle
        // densities, tile/team/osub/z_batches.  Substantially more
        // expensive than sweep() (tens of seconds to a few minutes on a
        // GPU); intended for opt-in via IPPL_AUTO_TUNE=full when the user
        // wants a CSV that's actually optimised for this machine.
        //
        // Each (method, width, density bucket) gets its OWN row in the CSV,
        // so the density-aware lookup in TileSizeCache picks the closest
        // recorded rho at runtime.
        // ====================================================================
        template <typename ExecSpace>
        std::vector<Sample> sweep_full() {
            std::vector<Sample> out;

            const bool host_backend =
#ifdef KOKKOS_ENABLE_OPENMP
                std::is_same_v<ExecSpace, Kokkos::OpenMP>
#else
                false
#endif
                ;

            // Grid sizes to probe — small / medium / large fits typical PIC
            // working sets without turning the sweep into a benchmark suite.
            const std::vector<unsigned> grids =
                host_backend ? std::vector<unsigned>{32, 64}
                             : std::vector<unsigned>{32, 64, 128};

            // Particle-per-cell densities to bucket the cache by.
            const std::vector<double> rhos = {0.5, 2.0, 8.0, 32.0};

            // Candidate sets per method.
            const std::vector<Vector<int, 3>> tiled_tiles =
                host_backend
                    ? std::vector<Vector<int, 3>>{{2, 2, 2}, {4, 4, 4}, {8, 8, 8}}
                    : std::vector<Vector<int, 3>>{
                          {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}, {6, 6, 6}, {8, 8, 8}};
            const std::vector<int> tiled_teams =
                host_backend ? std::vector<int>{1} : std::vector<int>{32, 64, 128, 256};
            const std::vector<int> tiled_osubs =
                host_backend ? std::vector<int>{1} : std::vector<int>{1, 2, 4};

            const std::vector<Vector<int, 3>> of_tiles =
                host_backend
                    ? std::vector<Vector<int, 3>>{{2, 2, 2}, {4, 4, 4}}
                    : std::vector<Vector<int, 3>>{
                          {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}, {6, 6, 6}};
            const std::vector<int> of_teams =
                host_backend ? std::vector<int>{1} : std::vector<int>{64, 128, 256, 512};
            const std::vector<int> of_zbs    = {1, 2, 4, 8};
            const std::vector<int> of_osubs  =
                host_backend ? std::vector<int>{1} : std::vector<int>{1, 2, 4};

            // Kernel widths covered by the runtime cache (CIC + NGP).
            const std::vector<int> widths = {1, 2};

            const int default_team = host_backend ? 1 : 32;
            const int meas_runs    = host_backend ? 5 : 7;

            const bool is_rank_zero = (ippl::Comm == nullptr) || (ippl::Comm->rank() == 0);

            // Total number of configs we'll time, for progress reporting.
            const size_t total_configs = grids.size() * rhos.size() * widths.size() * (
                /*Atomic*/ 1
                + tiled_tiles.size() * tiled_teams.size() * tiled_osubs.size()
                + of_tiles.size() * of_teams.size() * of_zbs.size() * of_osubs.size()
            );
            size_t done = 0;

            auto progress = [&](const char* label) {
                if (is_rank_zero && ippl::Info && (done % 16 == 0 || done == total_configs)) {
                    *ippl::Info << ::level1
                                << "[AutoTune-full] " << done << " / " << total_configs
                                << " configs measured (" << label << ")" << endl;
                }
            };

            for (unsigned N : grids) {
                for (double rho : rhos) {
                    const size_t nParticle = std::max<size_t>(
                        1, static_cast<size_t>(rho * double(N) * double(N) * double(N)));

                    for (int w : widths) {
                        // Atomic — only team_size matters; tile is irrelevant.
                        {
                            Vector<int, 3> tile{1, 1, 1};
                            const double tp = time_config<ExecSpace>(
                                ScatterMethod::Atomic, tile, default_team, 1, 1,
                                N, nParticle, meas_runs);
                            out.push_back(Sample{ScatterMethod::Atomic, "real", w, rho,
                                                 1, 1, 1, default_team, 1, 1, tp, 0.0});
                            ++done;
                            progress("Atomic");
                        }

                        // Tiled — sweep tile × team × osub.
                        Sample best_tiled{ScatterMethod::Tiled, "real", w, rho, 4, 4, 4,
                                          host_backend ? 1 : 64, 1, 1, 0.0, 0.0};
                        for (const auto& t : tiled_tiles) {
                            for (int team : tiled_teams) {
                                for (int osub : tiled_osubs) {
                                    const double tp = time_config<ExecSpace>(
                                        ScatterMethod::Tiled, t, team, osub, 1,
                                        N, nParticle, meas_runs);
                                    if (tp > best_tiled.throughput_Mpts_s) {
                                        best_tiled = Sample{ScatterMethod::Tiled, "real", w, rho,
                                                            t[0], t[1], t[2],
                                                            team, osub, 1, tp, 0.0};
                                    }
                                    ++done;
                                    progress("Tiled");
                                }
                            }
                        }
                        out.push_back(best_tiled);

                        // OutputFocused — sweep tile × team × osub × z_batches.
                        Sample best_of{ScatterMethod::OutputFocused, "real", w, rho, 2, 2, 2,
                                       host_backend ? 1 : 128, 1, 1, 0.0, 0.0};
                        for (const auto& t : of_tiles) {
                            for (int team : of_teams) {
                                for (int zb : of_zbs) {
                                    for (int osub : of_osubs) {
                                        const double tp = time_config<ExecSpace>(
                                            ScatterMethod::OutputFocused, t, team, osub, zb,
                                            N, nParticle, meas_runs);
                                        if (tp > best_of.throughput_Mpts_s) {
                                            best_of = Sample{
                                                ScatterMethod::OutputFocused, "real", w, rho,
                                                t[0], t[1], t[2], team, osub, zb, tp, 0.0};
                                        }
                                        ++done;
                                        progress("OutputFocused");
                                    }
                                }
                            }
                        }
                        out.push_back(best_of);
                    }
                }
            }

            // Mirror real → complex (complex scatter has the same access
            // pattern, only the value type differs; tuning is overwhelmingly
            // dominated by memory bandwidth which is the same).
            const size_t base = out.size();
            for (size_t i = 0; i < base; ++i) {
                Sample s     = out[i];
                s.value_type = "complex";
                out.push_back(s);
            }
            return out;
        }

        template <typename ExecSpace>
        GatherSample sweep_gather_full() {
            GatherSample best{GatherMethod::Atomic, 1, 1, 1, 0.0};

            const bool host_backend =
#ifdef KOKKOS_ENABLE_OPENMP
                std::is_same_v<ExecSpace, Kokkos::OpenMP>
#else
                false
#endif
                ;

            const std::vector<unsigned> grids =
                host_backend ? std::vector<unsigned>{32, 64}
                             : std::vector<unsigned>{32, 64, 128};
            const std::vector<double> rhos = {0.5, 2.0, 8.0, 32.0};
            const std::vector<Vector<int, 3>> sort_tiles =
                host_backend
                    ? std::vector<Vector<int, 3>>{{4, 4, 4}, {8, 8, 8}}
                    : std::vector<Vector<int, 3>>{
                          {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {6, 6, 6}, {8, 8, 8}};

            for (unsigned N : grids) {
                for (double rho : rhos) {
                    const size_t nParticle = std::max<size_t>(
                        1, static_cast<size_t>(rho * double(N) * double(N) * double(N)));

                    const double atomic_tp = time_gather<ExecSpace>(
                        GatherMethod::Atomic, {1, 1, 1}, N, nParticle, host_backend ? 5 : 7);
                    if (atomic_tp > best.throughput_Mpts_s) {
                        best = GatherSample{GatherMethod::Atomic, 1, 1, 1, atomic_tp};
                    }
                    for (const auto& t : sort_tiles) {
                        const double tp = time_gather<ExecSpace>(
                            GatherMethod::AtomicSort, t, N, nParticle, host_backend ? 5 : 7);
                        if (tp > best.throughput_Mpts_s) {
                            best =
                                GatherSample{GatherMethod::AtomicSort, t[0], t[1], t[2], tp};
                        }
                    }
                }
            }
            return best;
        }

    }  // namespace

    void seedBuiltinDefaults() {
        // Touch instances first so their lazy load() runs and any CSV in
        // cwd / IPPL_TILE_CSV / IPPL_GATHER_CSV gets honoured. Only seed if
        // nothing was loaded.
        auto& tcache = TileSizeCache::instance();
        auto& gcache = GatherCache::instance();

        const bool tcache_empty = !tcache.loaded();
        const bool gcache_empty = !gcache.get().has_value();
        if (!tcache_empty && !gcache_empty) {
            return;
        }

        // Pick a backend label and per-method recipes. These match what the
        // sweep would pick on a typical NVIDIA Ampere-class GPU and on a
        // multi-core OpenMP host; they are deliberately conservative so
        // they're never *bad*, just maybe not optimal for an unusual GPU.
        // Users who want machine-specific tuning can opt in with
        // IPPL_AUTO_TUNE=1.
        const char* backend = "Serial";
        int  atomic_team    = 1;
        int  tiled_team     = 1, tiled_tile  = 1;
        int  of_team        = 1, of_tile     = 1, of_zb       = 1;
        bool seed_tiled_of  = false;
        bool gather_sort    = false;
        int  gather_tile    = 1;

#ifdef KOKKOS_ENABLE_CUDA
        if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>) {
            backend        = "Kokkos::Cuda";
            atomic_team    = 32;
            tiled_team     = 64;  tiled_tile = 4;
            of_team        = 128; of_tile    = 4;  of_zb = 1;
            seed_tiled_of  = true;
            gather_sort    = true;
            gather_tile    = 4;
        }
#endif
#ifdef KOKKOS_ENABLE_HIP
        if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>) {
            backend        = "Kokkos::HIP";
            // CDNA wavefronts are 64-wide; favour larger team sizes than
            // CUDA's warp-32. Numbers are conservative starting points;
            // run IPPL_AUTO_TUNE=full and commit the resulting CSV to
            // cmake/auto_tune/<gfx*>/ for true machine-tuned values.
            atomic_team    = 64;
            tiled_team     = 128; tiled_tile = 4;
            of_team        = 256; of_tile    = 4;  of_zb = 1;
            seed_tiled_of  = true;
            gather_sort    = true;
            gather_tile    = 4;
        }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
        if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>) {
            backend        = "Kokkos::OpenMP";
            // OpenMP scatter is fastest with the Atomic kernel; Tiled and
            // OutputFocused incur thread-local-bin overhead that doesn't
            // pay off without massive parallelism.
            atomic_team    = 1;
            seed_tiled_of  = false;
            gather_sort    = false;
            gather_tile    = 1;
        }
#endif

        const bool is_rank_zero = (ippl::Comm == nullptr) || (ippl::Comm->rank() == 0);
        if (is_rank_zero && ippl::Info) {
            *ippl::Info << ::level2
                        << "[AutoTune] seeding built-in scatter/gather defaults for "
                        << backend << " (set IPPL_AUTO_TUNE=1 to run the sweep instead)"
                        << endl;
        }

        if (tcache_empty) {
            // throughput=1.0 is just a sentinel so the seeded entry compares
            // sensibly if a CSV later overrides it (CSV will have a real
            // throughput). Width 1 and 2 cover the PIC kernels in use.
            for (int w : {1, 2}) {
                for (bool cx : {false, true}) {
                    {
                        TileCacheEntry e;
                        e.tile.fill(1);
                        e.team_size               = atomic_team;
                        e.oversubscription_factor = 1;
                        e.z_batches               = 1;
                        e.is_rectangular          = false;
                        e.throughput_Mpts_s       = 1.0;
                        tcache.seed_default(ScatterMethod::Atomic, w, cx, e);
                    }
                    if (seed_tiled_of) {
                        {
                            TileCacheEntry e;
                            e.tile.fill(tiled_tile);
                            e.team_size               = tiled_team;
                            e.oversubscription_factor = 1;
                            e.z_batches               = 1;
                            e.is_rectangular          = false;
                            e.throughput_Mpts_s       = 1.0;
                            tcache.seed_default(ScatterMethod::Tiled, w, cx, e);
                        }
                        {
                            TileCacheEntry e;
                            e.tile.fill(of_tile);
                            e.team_size               = of_team;
                            e.oversubscription_factor = 1;
                            e.z_batches               = of_zb;
                            e.is_rectangular          = false;
                            e.throughput_Mpts_s       = 1.0;
                            tcache.seed_default(ScatterMethod::OutputFocused, w, cx, e);
                        }
                    }
                }
            }
        }

        if (gcache_empty) {
            gcache.seed_default(gather_sort ? GatherMethod::AtomicSort : GatherMethod::Atomic,
                                {gather_tile, gather_tile, gather_tile});
        }
    }

    bool runOnFirstUse(const std::string& output_path) {
        // Opt-in only. Default is to skip the sweep entirely and rely on the
        // built-in defaults seeded into TileSizeCache / GatherCache by
        // ippl::initialize.
        //
        //   IPPL_AUTO_TUNE=1     → quick sweep (~seconds)
        //   IPPL_AUTO_TUNE=full  → full sweep (tens of seconds to minutes;
        //                          much broader candidate set, multiple grid
        //                          sizes and densities, longer measurement)
        //   anything else / unset → no-op
        const char* enable = std::getenv("IPPL_AUTO_TUNE");
        if (enable == nullptr) {
            return false;
        }
        const std::string mode(enable);
        const bool quick_mode = (mode == "1" || mode == "quick");
        const bool full_mode  = (mode == "full" || mode == "2");
        if (!quick_mode && !full_mode) {
            return false;
        }

        const std::string gather_path =
            std::filesystem::path(output_path).replace_filename("gather_sweep_optimal.csv").string();

        const bool scatter_done = std::filesystem::exists(output_path);
        const bool gather_done  = std::filesystem::exists(gather_path);
        if (scatter_done && gather_done) {
            if (ippl::Info && (ippl::Comm == nullptr || ippl::Comm->rank() == 0)) {
                *ippl::Info << ::level1 << "[AutoTune] reusing existing sweep CSVs ("
                            << output_path << ", " << gather_path << ")" << endl;
            }
            return true;
        }

        const bool is_rank_zero = (ippl::Comm == nullptr) || (ippl::Comm->rank() == 0);

        if (is_rank_zero && ippl::Info) {
            *ippl::Info << ::level1
                        << "[AutoTune] IPPL_AUTO_TUNE=" << mode
                        << " — running " << (full_mode ? "FULL" : "quick")
                        << " width-2 scatter/gather sweep "
                        << (full_mode ? "(can take minutes; broader candidate set across grids "
                                        "and densities)"
                                      : "(this can take a few seconds; "
                                        "set IPPL_AUTO_TUNE=full for a deeper search)")
                        << endl;
        }

        [[maybe_unused]] auto run_scatter = [&](auto&& sweep_fn, const char* label) {
            if (ippl::Info) {
                *ippl::Info << ::level1
                            << "[AutoTune]   sweeping scatter on " << label << " → "
                            << output_path << endl;
            }
            write_csv(output_path, sweep_fn());
        };
        [[maybe_unused]] auto run_gather = [&](auto&& sweep_fn, const char* label) {
            if (ippl::Info) {
                *ippl::Info << ::level1
                            << "[AutoTune]   sweeping gather on " << label << " → "
                            << gather_path << endl;
            }
            write_gather_csv(gather_path, sweep_fn());
        };

        if (is_rank_zero) {
#ifdef KOKKOS_ENABLE_CUDA
            if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>) {
                if (!scatter_done) {
                    if (full_mode) run_scatter([] { return sweep_full<Kokkos::Cuda>(); },
                                               "Kokkos::Cuda (full)");
                    else           run_scatter([] { return sweep<Kokkos::Cuda>(); },
                                               "Kokkos::Cuda");
                }
                if (!gather_done) {
                    if (full_mode) run_gather([] { return sweep_gather_full<Kokkos::Cuda>(); },
                                              "Kokkos::Cuda (full)");
                    else           run_gather([] { return sweep_gather<Kokkos::Cuda>(); },
                                              "Kokkos::Cuda");
                }
            }
#endif
#ifdef KOKKOS_ENABLE_HIP
            if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>) {
                if (!scatter_done) {
                    if (full_mode) run_scatter([] { return sweep_full<Kokkos::HIP>(); },
                                               "Kokkos::HIP (full)");
                    else           run_scatter([] { return sweep<Kokkos::HIP>(); },
                                               "Kokkos::HIP");
                }
                if (!gather_done) {
                    if (full_mode) run_gather([] { return sweep_gather_full<Kokkos::HIP>(); },
                                              "Kokkos::HIP (full)");
                    else           run_gather([] { return sweep_gather<Kokkos::HIP>(); },
                                              "Kokkos::HIP");
                }
            }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
            if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>) {
                if (!scatter_done) {
                    if (full_mode) run_scatter([] { return sweep_full<Kokkos::OpenMP>(); },
                                               "Kokkos::OpenMP (full)");
                    else           run_scatter([] { return sweep<Kokkos::OpenMP>(); },
                                               "Kokkos::OpenMP");
                }
                if (!gather_done) {
                    if (full_mode) run_gather([] { return sweep_gather_full<Kokkos::OpenMP>(); },
                                              "Kokkos::OpenMP (full)");
                    else           run_gather([] { return sweep_gather<Kokkos::OpenMP>(); },
                                              "Kokkos::OpenMP");
                }
            }
#endif
            // Fallback for backends with nothing to tune (Serial, etc.).
            const bool serial_backend =
                true
#ifdef KOKKOS_ENABLE_CUDA
                && !std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>
#endif
#ifdef KOKKOS_ENABLE_HIP
                && !std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>
#endif
#ifdef KOKKOS_ENABLE_OPENMP
                && !std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>
#endif
                ;
            if (serial_backend) {
                if (!scatter_done) write_trivial(output_path);
                if (!gather_done)  write_trivial_gather(gather_path);
            }
        }
        if (ippl::Comm != nullptr) {
            ippl::Comm->barrier();
        }

        const bool ok = std::filesystem::exists(output_path);
        if (ok && is_rank_zero && ippl::Info) {
            *ippl::Info << ::level1 << "[AutoTune] sweep complete." << endl;
        }
        return ok;
    }

}  // namespace ippl::Interpolation::AutoTune
