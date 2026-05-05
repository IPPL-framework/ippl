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
#include <Kokkos_Random.hpp>

#include "Utility/GPModel.h"

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

        // Per-method per-backend cap on team_size, accounting for the inner
        // vector_length each scatter functor uses. GPUs limit a single team to
        // 1024 hardware threads (HIP enforces strictly less-than 1024); blowing
        // this cap during a sweep aborts the whole run, so we filter out
        // infeasible candidates before timing them.
        template <typename ExecSpace>
        int max_team_size_for(ScatterMethod m) {
            constexpr bool is_cuda =
#ifdef KOKKOS_ENABLE_CUDA
                std::is_same_v<ExecSpace, Kokkos::Cuda>;
#else
                false;
#endif
            constexpr bool is_hip =
#ifdef KOKKOS_ENABLE_HIP
                std::is_same_v<ExecSpace, Kokkos::HIP>;
#else
                false;
#endif
            if (!is_cuda && !is_hip) return 1;

            // Mirrors AtomicScatter::vector_length; Tiled and OutputFocused
            // construct their TeamPolicy with vector_length=1.
            const int vlen = (m == ScatterMethod::Atomic) ? (is_hip ? 64 : 32) : 1;
            const int hw   = is_hip ? 1023 : 1024;
            return std::max(1, hw / vlen);
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

            // Device-side RNG
            {
                using PoolType = Kokkos::Random_XorShift64_Pool<ExecSpace>;
                PoolType pool(static_cast<uint64_t>(42 + ippl::Comm->rank()));
                auto Rview = bunch.R.getView();
                Kokkos::parallel_for(
                    "AutoTune::initR",
                    Kokkos::RangePolicy<ExecSpace>(0, nLoc),
                    KOKKOS_LAMBDA(const size_t i) {
                        auto gen = pool.get_state();
                        Rview(i)[0] = gen.drand(0.01, 0.99);
                        Rview(i)[1] = gen.drand(0.01, 0.99);
                        Rview(i)[2] = gen.drand(0.01, 0.99);
                        pool.free_state(gen);
                    });
                Kokkos::fence();
            }
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


        struct ThroughputStats {
            double mean         = 0.0;  // Mpts/s
            double stddev       = 0.0;  // Mpts/s
            double mean_time_ms = 0.0;
            int    n_runs       = 0;
        };

        template <typename ExecSpace>
        ThroughputStats time_config_stats(ScatterMethod method,
                                          const Vector<int, 3>& tile_size, int team_size,
                                          int osub, int z_batches, unsigned N = 32,
                                          size_t nParticle = 100000, int runs = 5) {
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

            {
                using PoolType = Kokkos::Random_XorShift64_Pool<ExecSpace>;
                PoolType pool(static_cast<uint64_t>(42 + ippl::Comm->rank()));
                auto Rview = bunch.R.getView();
                Kokkos::parallel_for(
                    "AutoTune::initR",
                    Kokkos::RangePolicy<ExecSpace>(0, nLoc),
                    KOKKOS_LAMBDA(const size_t i) {
                        auto gen = pool.get_state();
                        Rview(i)[0] = gen.drand(0.01, 0.99);
                        Rview(i)[1] = gen.drand(0.01, 0.99);
                        Rview(i)[2] = gen.drand(0.01, 0.99);
                        pool.free_state(gen);
                    });
                Kokkos::fence();
            }
            bunch.Q = 1.0;
            bunch.update();

            ScatterConfig<3> cfg = ScatterConfig<3>::template get_default<ExecSpace>();
            cfg.method           = method;
            cfg.set_tile_size(tile_size);
            if (team_size > 0) cfg.team_size = team_size;
            if (osub > 0) cfg.oversubscription_factor = osub;
            if (z_batches > 0) cfg.z_batches = z_batches;
            cfg.lock_method   = true;
            cfg.enable_tuning = false;

            ippl::Interpolation::LinearKernel<value_t> cic;

            // Warm up.
            field = 0.0;
            try {
                bunch.Q.scatter_kernel(field, bunch.R, cic, cfg);
            } catch (...) {
                return {};
            }
            Kokkos::fence();

            std::vector<double> times_ms;
            times_ms.reserve(runs);
            for (int r = 0; r < runs; ++r) {
                field = 0.0;
                Kokkos::fence();
                auto t0 = std::chrono::steady_clock::now();
                try {
                    bunch.Q.scatter_kernel(field, bunch.R, cic, cfg);
                } catch (...) {
                    return {};
                }
                Kokkos::fence();
                auto t1 = std::chrono::steady_clock::now();
                times_ms.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            ThroughputStats out;
            out.n_runs       = runs;
            double sum_ms    = 0.0;
            for (double t : times_ms) sum_ms += t;
            out.mean_time_ms = sum_ms / std::max(1, runs);
            // Throughput per run, then mean+stddev so noise lives in throughput units.
            std::vector<double> tps;
            tps.reserve(runs);
            for (double t : times_ms) {
                tps.push_back(t > 0 ? double(nLoc) / 1e6 / (t * 1e-3) : 0.0);
            }
            double mean = 0;
            for (double v : tps) mean += v;
            mean /= std::max<int>(1, runs);
            double var = 0;
            for (double v : tps) var += (v - mean) * (v - mean);
            var /= std::max<int>(1, runs);
            out.mean   = mean;
            out.stddev = std::sqrt(var);
            return out;
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

            {
                using PoolType = Kokkos::Random_XorShift64_Pool<ExecSpace>;
                PoolType pool(static_cast<uint64_t>(42 + ippl::Comm->rank()));
                auto Rview = bunch.R.getView();
                Kokkos::parallel_for(
                    "AutoTune::initR",
                    Kokkos::RangePolicy<ExecSpace>(0, nLoc),
                    KOKKOS_LAMBDA(const size_t i) {
                        auto gen = pool.get_state();
                        Rview(i)[0] = gen.drand(0.01, 0.99);
                        Rview(i)[1] = gen.drand(0.01, 0.99);
                        Rview(i)[2] = gen.drand(0.01, 0.99);
                        pool.free_state(gen);
                    });
                Kokkos::fence();
            }
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
            // Cap each method's team size at what the backend actually
            // supports (HIP rejects team_size*vector_length >= 1024; on AMD
            // with Atomic's vlen=64 this bites hard).
            const int atomic_cap = max_team_size_for<ExecSpace>(ScatterMethod::Atomic);
            const int tiled_cap  = max_team_size_for<ExecSpace>(ScatterMethod::Tiled);
            const int of_cap     = max_team_size_for<ExecSpace>(ScatterMethod::OutputFocused);

            const int atomic_team = host_backend ? 1 : std::min(32, atomic_cap);
            const int tiled_team  = host_backend ? 1 : std::min(64, tiled_cap);
            const int of_team     = host_backend ? 1 : std::min(128, of_cap);

            // Atomic — tile is irrelevant to the dispatcher; fix to (1,1,1).
            {
                Vector<int, 3> tile{1, 1, 1};
                const double tp =
                    time_config<ExecSpace>(ScatterMethod::Atomic, tile, atomic_team, 1, 1);
                out.push_back(Sample{ScatterMethod::Atomic, "real", 2, 0.0,
                                     1, 1, 1, atomic_team, 1, 1, tp, 0.0});
            }

            // Tiled — small candidate set over (tile, team, osub).
            {
                Sample best{ScatterMethod::Tiled, "real", 2, 0.0, 4, 4, 4,
                            tiled_team, 1, 1, 0.0, 0.0};
                for (const auto& t :
                     std::vector<Vector<int, 3>>{{2, 2, 2}, {4, 4, 4}, {8, 8, 8}}) {
                    const double tp = time_config<ExecSpace>(ScatterMethod::Tiled, t,
                                                             tiled_team, 1, 1);
                    if (tp > best.throughput_Mpts_s) {
                        best = Sample{ScatterMethod::Tiled, "real", 2, 0.0, t[0], t[1], t[2],
                                      tiled_team, 1, 1, tp, 0.0};
                    }
                }
                out.push_back(best);
            }

            // OutputFocused — tile + z_batches.
            {
                Sample best{ScatterMethod::OutputFocused, "real", 2, 0.0, 2, 2, 2,
                            of_team, 1, 1, 0.0, 0.0};
                for (const auto& t : std::vector<Vector<int, 3>>{{2, 2, 2}, {4, 4, 4}}) {
                    for (int zb : {1, 4}) {
                        const double tp = time_config<ExecSpace>(
                            ScatterMethod::OutputFocused, t, of_team, 1, zb);
                        if (tp > best.throughput_Mpts_s) {
                            best = Sample{ScatterMethod::OutputFocused, "real", 2, 0.0,
                                          t[0], t[1], t[2],
                                          of_team, 1, zb, tp, 0.0};
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
            // Filter team-size candidates against per-backend caps. On HIP,
            // Atomic's vlen=64 caps team_size at 15 (1023/64), which would
            // discard every Tiled candidate above 15 if reused as-is — but
            // Tiled/OutputFocused use vlen=1, so they have a much higher cap.
            const int tiled_cap = max_team_size_for<ExecSpace>(ScatterMethod::Tiled);
            const int of_cap    = max_team_size_for<ExecSpace>(ScatterMethod::OutputFocused);
            const int atomic_cap = max_team_size_for<ExecSpace>(ScatterMethod::Atomic);

            auto filter_teams = [](std::vector<int> teams, int cap) {
                std::vector<int> out;
                for (int t : teams) if (t <= cap) out.push_back(t);
                if (out.empty()) out.push_back(std::max(1, cap));
                return out;
            };

            const std::vector<int> tiled_teams = filter_teams(
                host_backend ? std::vector<int>{1} : std::vector<int>{32, 64, 128, 256},
                tiled_cap);
            const std::vector<int> tiled_osubs =
                host_backend ? std::vector<int>{1} : std::vector<int>{1, 2, 4};

            const std::vector<Vector<int, 3>> of_tiles =
                host_backend
                    ? std::vector<Vector<int, 3>>{{2, 2, 2}, {4, 4, 4}}
                    : std::vector<Vector<int, 3>>{
                          {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}, {6, 6, 6}};
            const std::vector<int> of_teams = filter_teams(
                host_backend ? std::vector<int>{1} : std::vector<int>{64, 128, 256, 512},
                of_cap);
            const std::vector<int> of_zbs    = {1, 2, 4, 8};
            const std::vector<int> of_osubs  =
                host_backend ? std::vector<int>{1} : std::vector<int>{1, 2, 4};

            // Kernel widths covered by the runtime cache (CIC + NGP).
            const std::vector<int> widths = {1, 2};

            const int default_team = host_backend ? 1 : std::min(32, atomic_cap);
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
                                          host_backend ? 1 : std::min(64, tiled_cap),
                                          1, 1, 0.0, 0.0};
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
                                       host_backend ? 1 : std::min(128, of_cap),
                                       1, 1, 0.0, 0.0};
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

        // ====================================================================
        // Bayesian-Optimization-driven scatter sweep (IPPL_AUTO_TUNE=bo).
        //
        // For each (method, kernel_width, density) bucket we run a 5D BO over
        //   [tile_x, tile_y, tile_z, ts_idx, osub|z_batches]
        // using the heteroscedastic ARD-RBF GP in src/Utility/GPModel.h.
        //
        // Budget per bucket is controlled by IPPL_BO_BUDGET (default 60).
        // ====================================================================

        struct BOSearchPoint {
            // [0..2] = tile_x/y/z, [3] = ts_idx, [4] = osub or z_batches
            std::array<int, 5> v{};
            bool operator==(const BOSearchPoint& o) const noexcept { return v == o.v; }
        };

        struct BOSearchPointHash {
            std::size_t operator()(const BOSearchPoint& p) const noexcept {
                std::size_t h = 1469598103934665603ull;
                for (int x : p.v) {
                    h ^= static_cast<std::size_t>(x + 1000);
                    h *= 1099511628211ull;
                }
                return h;
            }
        };

        // Cap team_size given vector_length used by each method's TeamPolicy
        template <typename ExecSpace>
        std::vector<int> bo_team_size_candidates(ScatterMethod m, bool host_backend) {
            const int cap = max_team_size_for<ExecSpace>(m);
            std::vector<int> raw;
            if (host_backend) {
                raw = {1};
            } else if (m == ScatterMethod::Atomic) {
                raw = {1, 2, 4, 8, 16, 32, 64};
            } else if (m == ScatterMethod::Tiled) {
                raw = {16, 32, 64, 128, 256};
            } else {  // OutputFocused
                raw = {64, 128, 256, 512};
            }
            std::vector<int> out;
            for (int t : raw) {
                if (t <= cap) out.push_back(t);
            }
            if (out.empty()) out.push_back(std::max(1, cap));
            return out;
        }

        // Per-bucket BO. Returns the best-config sample (throughput in Mpts/s)
        // for (method, width, rho) at value_type=="real"
        template <typename ExecSpace>
        Sample sweep_bo_bucket(ScatterMethod method, int width, double rho,
                               unsigned N, size_t /*nParticle_outer*/, int budget,
                               bool host_backend) {
            const auto ts_cands = bo_team_size_candidates<ExecSpace>(method, host_backend);
            const int  hi_ts    = static_cast<int>(ts_cands.size()) - 1;
            const int  lo_tile = 1;
            const int  hi_tile = host_backend ? 8 : 8;
            const bool uses_zb = (method == ScatterMethod::OutputFocused);
            const int  lo4     = host_backend ? 1 : (uses_zb ? 1 : 1);
            const int  hi4     = host_backend ? 1 : (uses_zb ? 8 : 4);

            using GP = ippl::detail::GPModel<5>;
            GP gp;
            GP::Point lo_gp{static_cast<double>(lo_tile), static_cast<double>(lo_tile),
                            static_cast<double>(lo_tile), 0.0,
                            static_cast<double>(lo4)};
            GP::Point hi_gp{static_cast<double>(hi_tile), static_cast<double>(hi_tile),
                            static_cast<double>(hi_tile), static_cast<double>(hi_ts),
                            static_cast<double>(hi4)};
            gp.set_bounds(lo_gp, hi_gp);

            std::unordered_map<BOSearchPoint, double, BOSearchPointHash> tp_cache;
            std::unordered_map<BOSearchPoint, double, BOSearchPointHash> tpvar_cache;

            // Atomic only varies team_size; collapse the search dims so the GP
            // sees only ts_idx as informative (other dims pinned to 1).
            const bool atomic = (method == ScatterMethod::Atomic);

            auto build_cfg_args = [&](const BOSearchPoint& p) {
                Vector<int, 3> tile;
                if (atomic) {
                    tile = {1, 1, 1};
                } else {
                    tile = {p.v[0], p.v[1], p.v[2]};
                }
                int team = ts_cands[std::clamp(p.v[3], 0, hi_ts)];
                int osub = uses_zb ? 1 : p.v[4];
                int zb   = uses_zb ? p.v[4] : 1;
                return std::tuple<Vector<int, 3>, int, int, int>{tile, team, osub, zb};
            };

            auto evaluate = [&](const BOSearchPoint& pt, unsigned grid_N,
                                int runs) -> ThroughputStats {
                auto it = tp_cache.find(pt);
                if (it != tp_cache.end()) {
                    ThroughputStats s;
                    s.mean   = it->second;
                    s.stddev = std::sqrt(tpvar_cache[pt]);
                    s.n_runs = runs;
                    return s;
                }
                auto [tile, team, osub, zb] = build_cfg_args(pt);
                const size_t np_eff =
                    std::max<size_t>(1, static_cast<size_t>(rho * double(grid_N)
                                                            * double(grid_N) * double(grid_N)));
                ThroughputStats s = time_config_stats<ExecSpace>(
                    method, tile, team, osub, zb, grid_N, np_eff, runs);
                tp_cache[pt]    = s.mean;
                tpvar_cache[pt] = s.stddev * s.stddev;
                return s;
            };

            std::mt19937 rng(0x9e3779b9u + width * 0x100 + static_cast<unsigned>(method));
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            auto rand_int = [&](int lo, int hi) {
                if (lo >= hi) return lo;
                return lo + static_cast<int>(u01(rng) * (hi - lo + 1));
            };

            BOSearchPoint best_pt{};
            best_pt.v   = {lo_tile, lo_tile, lo_tile, 0, std::max(lo4, 1)};
            double best = 0.0;

            const int meas_runs = host_backend ? 5 : 6;

            auto eval_and_record = [&](const BOSearchPoint& pt) {
                ThroughputStats s = evaluate(pt, N, meas_runs);
                if (s.mean > 0) {
                    GP::Point xp;
                    for (int d = 0; d < 5; ++d) {
                        xp[d] = static_cast<double>(pt.v[d]);
                    }
                    gp.add(xp, s.mean, s.stddev * s.stddev);
                }
                if (s.mean > best) {
                    best    = s.mean;
                    best_pt = pt;
                }
                return s;
            };

            // Phase 0: structured anchors. Always include the cube tiles the
            // grid sweep enumerated, paired with sensible team-size defaults,
            // so the BO never returns worse than the structured optimum and
            // the GP starts with informative observations along the cube line.
            const std::vector<int> anchor_cubes = {2, 3, 4, 5, 6, 8};
            // Pick a per-method default team-size index that lands near the
            // grid sweep's seed (atomic_team=32, tiled_team=64, of_team=128).
            auto closest_ts_idx = [&](int target) {
                int best_i = 0, best_d = INT_MAX;
                for (int i = 0; i <= hi_ts; ++i) {
                    int d = std::abs(ts_cands[i] - target);
                    if (d < best_d) { best_d = d; best_i = i; }
                }
                return best_i;
            };
            const int default_ts =
                atomic ? closest_ts_idx(32)
                       : (method == ScatterMethod::Tiled ? closest_ts_idx(64)
                                                         : closest_ts_idx(128));
            if (atomic) {
                // Atomic anchor: just sweep the team_size candidates with osub=1.
                for (int i = 0; i <= hi_ts; ++i) {
                    BOSearchPoint pt;
                    pt.v = {1, 1, 1, i, std::max(lo4, 1)};
                    eval_and_record(pt);
                }
            } else {
                for (int t : anchor_cubes) {
                    if (t < lo_tile || t > hi_tile) continue;
                    BOSearchPoint pt;
                    pt.v = {t, t, t, default_ts, std::max(lo4, 1)};
                    eval_and_record(pt);
                }
            }

            // Phase 1: LHS over the full 5D box (rectangular tiles, varied
            // team & osub/zb). Smaller now because Phase 0 already provides
            // structured coverage along the cube axis.
            const int n_init = std::max(6, std::min(budget / 4, 4 * (hi_ts + 1)));
            for (int i = 0; i < n_init; ++i) {
                BOSearchPoint pt;
                pt.v[0] = atomic ? 1 : rand_int(lo_tile, hi_tile);
                pt.v[1] = atomic ? 1 : rand_int(lo_tile, hi_tile);
                pt.v[2] = atomic ? 1 : rand_int(lo_tile, hi_tile);
                pt.v[3] = rand_int(0, hi_ts);
                pt.v[4] = rand_int(lo4, hi4);
                eval_and_record(pt);
            }
            if (gp.size() >= 4) {
                gp.fit();
            }

            // Phase 2: full-fidelity acquisition loop.
            const int n_phase0 = atomic ? (hi_ts + 1)
                                        : static_cast<int>(anchor_cubes.size());
            const int n_acq    = std::max(0, budget - n_phase0 - n_init);
            for (int it = 0; it < n_acq; ++it) {
                if (gp.size() >= 4 && (it % 5) == 0) {
                    gp.fit();
                }

                const double progress = double(it) / std::max(1, n_acq);
                const bool   use_ucb  = (u01(rng) < 0.3 * (1.0 - 0.9 * progress));
                const double beta_ucb = 2.0 * std::sqrt(1.0 - progress) + 0.5;

                BOSearchPoint next  = best_pt;
                double        bestA = -1e300;
                auto score = [&](const BOSearchPoint& c) {
                    if (tp_cache.count(c)) return;
                    GP::Point xp;
                    for (int d = 0; d < 5; ++d) {
                        xp[d] = static_cast<double>(c.v[d]);
                    }
                    const double acq = use_ucb ? gp.upper_confidence_bound(xp, beta_ucb)
                                               : gp.expected_improvement(xp, best);
                    if (acq > bestA) {
                        bestA = acq;
                        next  = c;
                    }
                };
                // Random candidates over the full feasible box.
                for (int s = 0; s < 1500; ++s) {
                    BOSearchPoint c;
                    c.v[0] = atomic ? 1 : rand_int(lo_tile, hi_tile);
                    c.v[1] = atomic ? 1 : rand_int(lo_tile, hi_tile);
                    c.v[2] = atomic ? 1 : rand_int(lo_tile, hi_tile);
                    c.v[3] = rand_int(0, hi_ts);
                    c.v[4] = rand_int(lo4, hi4);
                    score(c);
                }
                // Local neighbourhood along each axis.
                static const int kDeltas[] = {-2, -1, 1, 2};
                for (int d = 0; d < 5; ++d) {
                    if (atomic && d < 3) continue;
                    for (int delta : kDeltas) {
                        BOSearchPoint c = best_pt;
                        const int lo = (d < 3) ? lo_tile : (d == 3 ? 0 : lo4);
                        const int hi = (d < 3) ? hi_tile : (d == 3 ? hi_ts : hi4);
                        c.v[d]       = std::clamp(best_pt.v[d] + delta, lo, hi);
                        score(c);
                    }
                }
                if (bestA <= -1e200) break;

                ThroughputStats s = evaluate(next, N, meas_runs);
                if (s.mean > 0) {
                    GP::Point xp;
                    for (int d = 0; d < 5; ++d) {
                        xp[d] = static_cast<double>(next.v[d]);
                    }
                    gp.add(xp, s.mean, s.stddev * s.stddev);
                }
                if (s.mean > best) {
                    best    = s.mean;
                    best_pt = next;
                }
            }

            // Phase 3: hill-climb polish around the best so far.
            for (int polish = 0; polish < 2; ++polish) {
                bool improved = false;
                for (int d = 0; d < 5; ++d) {
                    if (atomic && d < 3) continue;
                    for (int delta : {-1, 1}) {
                        BOSearchPoint c = best_pt;
                        const int lo = (d < 3) ? lo_tile : (d == 3 ? 0 : lo4);
                        const int hi = (d < 3) ? hi_tile : (d == 3 ? hi_ts : hi4);
                        c.v[d]       = std::clamp(best_pt.v[d] + delta, lo, hi);
                        if (c == best_pt || tp_cache.count(c)) continue;
                        ThroughputStats s = evaluate(c, N, meas_runs);
                        if (s.mean > best) {
                            best     = s.mean;
                            best_pt  = c;
                            improved = true;
                        }
                    }
                }
                if (!improved) break;
            }

            // Final full-fidelity remeasurement of the chosen config.
            ThroughputStats final_s = evaluate(best_pt, N,
                                               host_backend ? 7 : 9);
            if (final_s.mean > best) best = final_s.mean;

            const auto [tile, team, osub, zb] = build_cfg_args(best_pt);
            return Sample{method,
                          "real",
                          width,
                          rho,
                          tile[0], tile[1], tile[2],
                          team,
                          osub,
                          zb,
                          best,
                          final_s.mean_time_ms};
        }

        template <typename ExecSpace>
        std::vector<Sample> sweep_bo() {
            std::vector<Sample> out;
            const bool host_backend =
#ifdef KOKKOS_ENABLE_OPENMP
                std::is_same_v<ExecSpace, Kokkos::OpenMP>
#else
                false
#endif
                ;

            const unsigned N = host_backend ? 64u : 128u;

            const std::vector<double> rhos   = {0.5, 2.0, 4.0, 8.0};
            const std::vector<int>    widths = {1, 2};
            const std::vector<ScatterMethod> methods = {ScatterMethod::Atomic,
                                                        ScatterMethod::Tiled,
                                                        ScatterMethod::OutputFocused};

            // Per-bucket budget. Tuned so total wall time is comparable to the
            // FULL grid sweep (a few minutes on GPU). User override via env.
            const char* budget_env = std::getenv("IPPL_BO_BUDGET");
            const int   per_bucket =
                budget_env ? std::max(20, std::atoi(budget_env)) : 60;

            const bool is_rank_zero =
                (ippl::Comm == nullptr) || (ippl::Comm->rank() == 0);

            const size_t total_buckets =
                rhos.size() * widths.size() * methods.size();
            size_t done = 0;

            for (double rho : rhos) {
                const size_t nParticle = std::max<size_t>(
                    1, static_cast<size_t>(rho * double(N) * double(N) * double(N)));
                for (int w : widths) {
                    for (ScatterMethod m : methods) {
                        ++done;
                        if (is_rank_zero && ippl::Info) {
                            *ippl::Info << ::level1 << "[AutoTune-bo] " << done
                                        << " / " << total_buckets << "  N=" << N
                                        << " rho=" << rho
                                        << " (nP=" << nParticle << ")"
                                        << " w=" << w << " "
                                        << method_name(m) << endl;
                        }
                        Sample s = sweep_bo_bucket<ExecSpace>(
                            m, w, rho, N, nParticle, per_bucket, host_backend);
                        out.push_back(s);
                    }
                }
            }
            // Mirror real → complex (same memory access pattern).
            const size_t base = out.size();
            for (size_t i = 0; i < base; ++i) {
                Sample s     = out[i];
                s.value_type = "complex";
                out.push_back(s);
            }
            return out;
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
            // CDNA wavefronts are 64-wide. AtomicScatter uses vector_length=64
            // on HIP, so atomic_team * 64 must be < 1024 → atomic_team ≤ 15.
            // Tiled and OutputFocused construct their TeamPolicy with
            // vector_length=1, so the per-team thread cap is 1024 there.
            atomic_team    = 8;
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
        //   IPPL_AUTO_TUNE=full  → full grid sweep (tens of seconds to minutes;
        //                          much broader candidate set, multiple grid
        //                          sizes and densities, longer measurement)
        //   IPPL_AUTO_TUNE=bo    → Bayesian-optimization-driven sweep with
        //                          ARD-RBF GP, multi-fidelity warm-up, and
        //                          rectangular tile search. Per-bucket budget
        //                          via IPPL_BO_BUDGET (default 60).
        //   anything else / unset → no-op
        const char* enable = std::getenv("IPPL_AUTO_TUNE");
        if (enable == nullptr) {
            return false;
        }
        const std::string mode(enable);
        const bool quick_mode = (mode == "1" || mode == "quick");
        const bool full_mode  = (mode == "full" || mode == "2");
        const bool bo_mode    = (mode == "bo" || mode == "BO");
        if (!quick_mode && !full_mode && !bo_mode) {
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
            const char* mode_label = bo_mode ? "BO" : (full_mode ? "FULL" : "quick");
            const char* descr =
                bo_mode ? "(Bayesian optimization with ARD-RBF GP; per-bucket budget via "
                          "IPPL_BO_BUDGET, default 60)"
                        : (full_mode
                               ? "(can take minutes; broader candidate set across grids "
                                 "and densities)"
                               : "(this can take a few seconds; set IPPL_AUTO_TUNE=full or "
                                 "IPPL_AUTO_TUNE=bo for a deeper search)");
            *ippl::Info << ::level1
                        << "[AutoTune] IPPL_AUTO_TUNE=" << mode << " — running "
                        << mode_label << " width-2 scatter/gather sweep " << descr << endl;
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
                    if (bo_mode)        run_scatter([] { return sweep_bo<Kokkos::Cuda>(); },
                                                    "Kokkos::Cuda (BO)");
                    else if (full_mode) run_scatter([] { return sweep_full<Kokkos::Cuda>(); },
                                                    "Kokkos::Cuda (full)");
                    else                run_scatter([] { return sweep<Kokkos::Cuda>(); },
                                                    "Kokkos::Cuda");
                }
                if (!gather_done) {
                    if (full_mode || bo_mode)
                        run_gather([] { return sweep_gather_full<Kokkos::Cuda>(); },
                                   bo_mode ? "Kokkos::Cuda (BO)" : "Kokkos::Cuda (full)");
                    else
                        run_gather([] { return sweep_gather<Kokkos::Cuda>(); },
                                   "Kokkos::Cuda");
                }
            }
#endif
#ifdef KOKKOS_ENABLE_HIP
            if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>) {
                if (!scatter_done) {
                    if (bo_mode)        run_scatter([] { return sweep_bo<Kokkos::HIP>(); },
                                                    "Kokkos::HIP (BO)");
                    else if (full_mode) run_scatter([] { return sweep_full<Kokkos::HIP>(); },
                                                    "Kokkos::HIP (full)");
                    else                run_scatter([] { return sweep<Kokkos::HIP>(); },
                                                    "Kokkos::HIP");
                }
                if (!gather_done) {
                    if (full_mode || bo_mode)
                        run_gather([] { return sweep_gather_full<Kokkos::HIP>(); },
                                   bo_mode ? "Kokkos::HIP (BO)" : "Kokkos::HIP (full)");
                    else
                        run_gather([] { return sweep_gather<Kokkos::HIP>(); },
                                   "Kokkos::HIP");
                }
            }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
            if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>) {
                if (!scatter_done) {
                    if (bo_mode)        run_scatter([] { return sweep_bo<Kokkos::OpenMP>(); },
                                                    "Kokkos::OpenMP (BO)");
                    else if (full_mode) run_scatter([] { return sweep_full<Kokkos::OpenMP>(); },
                                                    "Kokkos::OpenMP (full)");
                    else                run_scatter([] { return sweep<Kokkos::OpenMP>(); },
                                                    "Kokkos::OpenMP");
                }
                if (!gather_done) {
                    if (full_mode || bo_mode)
                        run_gather([] { return sweep_gather_full<Kokkos::OpenMP>(); },
                                   bo_mode ? "Kokkos::OpenMP (BO)" : "Kokkos::OpenMP (full)");
                    else
                        run_gather([] { return sweep_gather<Kokkos::OpenMP>(); },
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
