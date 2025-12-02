#include "Ippl.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <random>

#include "FFT/NUFFT/ESKernel.h"
#include "FFT/NUFFT/NUFFTUtilities.h"
#include "Interpolation/ScatterConfig.h"

// Kokkos random number generator
#include <Kokkos_Random.hpp>

template <class PLayout>
struct Bunch : public ippl::ParticleBase<PLayout> {
    using base_type    = ippl::ParticleBase<PLayout>;
    using real_type    = double;
    using complex_type = Kokkos::complex<real_type>;

    Bunch(PLayout& playout)
        : base_type(playout) {
        this->addAttribute(Q_scatter);
        this->addAttribute(Q_gather);
        this->addAttribute(Q_gather_tiled);
    }

    // particle values for scatter (input to scatter_kernel)
    ippl::ParticleAttrib<complex_type> Q_scatter;
    // particle values for gather (Atomic config)
    ippl::ParticleAttrib<complex_type> Q_gather;
    // particle values for gather (Tiled config)
    ippl::ParticleAttrib<complex_type> Q_gather_tiled;
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    int exit_code = EXIT_SUCCESS;
    sleep(10);

    {
        // ---------------- Types / aliases ----------------
        constexpr unsigned Dim = 3;
        using real_type        = double;
        using complex_type     = Kokkos::complex<real_type>;
        using ExecSpace        = Kokkos::DefaultExecutionSpace;
        using MemSpace         = ExecSpace::memory_space;

        using Mesh_t      = ippl::UniformCartesian<real_type, Dim>;
        using Centering_t = Mesh_t::DefaultCentering;
        using Field_t     = ippl::Field<complex_type, Dim, Mesh_t, Centering_t>;

        using PLayout_t = ippl::ParticleSpatialLayout<real_type, Dim>;
        using Bunch_t   = Bunch<PLayout_t>;

        // Kokkos random pool type
        using RandPoolType = Kokkos::Random_XorShift64_Pool<ExecSpace>;

        int rank   = ippl::Comm->rank();
        int nRanks = ippl::Comm->size();

        // ---------------- Grid / layout ------------------
        ippl::Vector<std::size_t, Dim> n_grid;
        n_grid[0] = 32;
        n_grid[1] = 32;
        n_grid[2] = 32;

        ippl::NDIndex<Dim> domain;
        for (unsigned d = 0; d < Dim; ++d) {
            domain[d] = ippl::Index(n_grid[d]);
        }

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        bool periodicBC = true;
        ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, domain, isParallel, periodicBC);

        ippl::Vector<real_type, Dim> origin;
        ippl::Vector<real_type, Dim> hx;

        for (unsigned d = 0; d < Dim; ++d) {
            origin[d] = 0;
            hx[d]     = 2.0 * M_PI / static_cast<real_type>(n_grid[d]);
        }

        Mesh_t mesh(domain, hx, origin);

        // ---------------- NUFFT kernel / ghosts ----------
        ippl::NUFFT::ESKernel<real_type> kernel(1e-6);
        const int nghost = (kernel.width()) / 2 + 1;

        Field_t grid_random(mesh, layout, nghost);
        Field_t grid_scattered_atomic(mesh, layout, nghost);
        Field_t grid_scattered_tiled(mesh, layout, nghost);

        // ---------------- Particle layout / bunch --------
        PLayout_t playout(layout, mesh);
        Bunch_t bunch(playout);

        bunch.setParticleBC(ippl::BC::PERIODIC);

        const std::size_t Ntotal = 100000;
        if (Ntotal % nRanks != 0) {
            if (rank == 0) {
                std::cerr << "Ntotal (" << Ntotal << ") must be divisible by #ranks ("
                          << nRanks << ")\n";
            }
            ippl::finalize();
            return EXIT_FAILURE;
        }

        const std::size_t nLoc = Ntotal / nRanks;
        bunch.create(nLoc);

        // ---------------- Random positions & particle values (GPU) ----------------
        // Get local spatial domain bounds
        const auto& lDom = layout.getLocalNDIndex();
        ippl::Vector<real_type, Dim> local_min, local_max;
        for (unsigned d = 0; d < Dim; ++d) {
            local_min[d] = origin[d] + lDom[d].first() * hx[d];
            local_max[d] = origin[d] + (lDom[d].last() + 1) * hx[d];
        }

        // Create random pool on device with rank-dependent seed
        RandPoolType rand_pool(42 + rank * 12345);

        // Get device views
        auto R_view  = bunch.R.getView();
        auto Qs_view = bunch.Q_scatter.getView();
        auto Qg_view = bunch.Q_gather.getView();
        auto Qgt_view = bunch.Q_gather_tiled.getView();

        // Copy bounds to device-accessible variables
        const real_type lmin0 = local_min[0], lmax0 = local_max[0];
        const real_type lmin1 = local_min[1], lmax1 = local_max[1];
        const real_type lmin2 = local_min[2], lmax2 = local_max[2];

        // Initialize particles on GPU
        Kokkos::parallel_for("InitParticles",
            Kokkos::RangePolicy<ExecSpace>(0, nLoc),
            KOKKOS_LAMBDA(const std::size_t i) {
                // Get thread-local random generator
                auto gen = rand_pool.get_state();

                // Generate random position within local domain
                ippl::Vector<real_type, Dim> r;
                r[0] = gen.drand(lmin0, lmax0);
                r[1] = gen.drand(lmin1, lmax1);
                r[2] = gen.drand(lmin2, lmax2);
                R_view(i) = r;

                // Box-Muller transform
                real_type u1 = gen.drand(1e-10, 1.0);
                real_type u2 = gen.drand(0.0, 1.0);
                real_type u3 = gen.drand(1e-10, 1.0);
                real_type u4 = gen.drand(0.0, 1.0);

                real_type re = Kokkos::sqrt(-2.0 * Kokkos::log(u1)) * Kokkos::cos(2.0 * M_PI * u2);
                real_type im = Kokkos::sqrt(-2.0 * Kokkos::log(u3)) * Kokkos::cos(2.0 * M_PI * u4);
                Qs_view(i) = complex_type(re, im);

                // Initialize gather values to zero
                Qg_view(i)  = complex_type(0.0, 0.0);
                Qgt_view(i) = complex_type(0.0, 0.0);

                // Return state to pool
                rand_pool.free_state(gen);
            }
        );
        Kokkos::fence();

        // ---------------- Random grid for gather ----------------------------
        auto grid_view = grid_random.getView();

        // Get local domain bounds (in GLOBAL indices)
        const int i_start = lDom[0].first();
        const int j_start = lDom[1].first();
        const int k_start = lDom[2].first();
        const int i_end   = lDom[0].last() + 1;
        const int j_end   = lDom[1].last() + 1;
        const int k_end   = lDom[2].last() + 1;

        const std::size_t ng0 = n_grid[0];
        const std::size_t ng1 = n_grid[1];
        const int ng = nghost;

        // Initialize grid on GPU using MDRangePolicy
        using mdrange_policy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;

        Kokkos::parallel_for("InitGrid",
            mdrange_policy({k_start, j_start, i_start}, {k_end, j_end, i_end}),
            KOKKOS_LAMBDA(const int k_global, const int j_global, const int i_global) {
                // Compute flat index in global grid for deterministic RNG seed
                std::size_t global_idx =
                    k_global * ng0 * ng1 +
                    j_global * ng0 +
                    i_global;

                Kokkos::Random_XorShift64<ExecSpace> gen(42 + global_idx * 2);

                real_type re = gen.drand(-1.0, 1.0);
                real_type im = gen.drand(-1.0, 1.0);

                int i_local = i_global - i_start;
                int j_local = j_global - j_start;
                int k_local = k_global - k_start;

                grid_view(i_local + ng, j_local + ng, k_local + ng) = complex_type(re, im);
            }
        );
        Kokkos::fence();

        grid_random.fillHalo();
        bunch.update();

        // ---------------- Scatter configs: Atomic & Tiled -------------------
        ippl::Interpolation::ScatterConfig cfg_atomic =
            ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
        cfg_atomic.method = ippl::Interpolation::ScatterMethod::Atomic;
        cfg_atomic.sort   = true;

        ippl::Interpolation::ScatterConfig cfg_tiled =
            ippl::Interpolation::ScatterConfig::get_default<ExecSpace>();
        cfg_tiled.method = ippl::Interpolation::ScatterMethod::Tiled;
        cfg_tiled.sort   = true;

        // ====================================================================
        // 1) Adjointness test for Atomic and Tiled
        // ====================================================================
        auto run_adjointness = [&](const ippl::Interpolation::ScatterConfig& cfg,
                                   const char* label,
                                   real_type& rel_err_out) {
            // Scatter: grid_scattered_atomic used as scratch here
            Field_t& grid_scattered = grid_scattered_atomic;
            grid_scattered          = complex_type(0.0, 0.0);

            bunch.Q_scatter.scatter_kernel(grid_scattered, bunch.R, kernel, cfg);
            grid_scattered.accumulateHalo();

            // Gather: reuse Q_gather as destination
            bunch.Q_gather = complex_type(0.0, 0.0);
            bunch.Q_gather.gather(grid_random, bunch.R, kernel, false, cfg);

            // Compute inner products on GPU using parallel reductions
            auto grid_sc_view = grid_scattered.getView();
            auto grid_rnd_view = grid_random.getView();
            auto Qs_view_final = bunch.Q_scatter.getView();
            auto Qg_view_final = bunch.Q_gather.getView();

            std::size_t n_local_i = lDom[0].length();
            std::size_t n_local_j = lDom[1].length();
            std::size_t n_local_k = lDom[2].length();

            // Left inner product: <S q, g> on grid
            real_type inner_left_re = 0.0, inner_left_im = 0.0;
            const int ng_local = nghost;

            Kokkos::parallel_reduce("InnerLeft",
                mdrange_policy({0, 0, 0},
                              {static_cast<int>(n_local_k),
                               static_cast<int>(n_local_j),
                               static_cast<int>(n_local_i)}),
                KOKKOS_LAMBDA(const int kk, const int jj, const int ii,
                             real_type& sum_re, real_type& sum_im) {
                    const complex_type Scq = grid_sc_view(ii + ng_local, jj + ng_local, kk + ng_local);
                    const complex_type g   = grid_rnd_view(ii + ng_local, jj + ng_local, kk + ng_local);
                    complex_type prod = Kokkos::conj(Scq) * g;
                    sum_re += prod.real();
                    sum_im += prod.imag();
                },
                inner_left_re, inner_left_im
            );

            // Right inner product: <q, G g> on particles
            real_type inner_right_re = 0.0, inner_right_im = 0.0;

            Kokkos::parallel_reduce("InnerRight",
                Kokkos::RangePolicy<ExecSpace>(0, nLoc),
                KOKKOS_LAMBDA(const std::size_t i, real_type& sum_re, real_type& sum_im) {
                    const complex_type q  = Qs_view_final(i);
                    const complex_type Gg = Qg_view_final(i);
                    complex_type prod = Kokkos::conj(q) * Gg;
                    sum_re += prod.real();
                    sum_im += prod.imag();
                },
                inner_right_re, inner_right_im
            );

            // MPI reduce
            real_type global_left_re, global_left_im;
            real_type global_right_re, global_right_im;

            ippl::Comm->reduce(inner_left_re,  global_left_re,  1, std::plus<real_type>());
            ippl::Comm->reduce(inner_left_im,  global_left_im,  1, std::plus<real_type>());
            ippl::Comm->reduce(inner_right_re, global_right_re, 1, std::plus<real_type>());
            ippl::Comm->reduce(inner_right_im, global_right_im, 1, std::plus<real_type>());

            if (rank == 0) {
                std::complex<real_type> L(global_left_re,  global_left_im);
                std::complex<real_type> R(global_right_re, global_right_im);

                real_type num   = std::abs(L - R);
                real_type denom = std::max<real_type>({std::abs(L), std::abs(R), 1.0});
                rel_err_out     = num / denom;

                std::cout << "Adjointness (" << label << "):\n";
                std::cout << "  <S q, g> = " << L << "\n";
                std::cout << "  <q, G g> = " << R << "\n";
                std::cout << "  rel_err  = " << rel_err_out << "\n";
            }
        };

        real_type rel_atomic = 0.0;
        real_type rel_tiled  = 0.0;
        run_adjointness(cfg_atomic, "Atomic", rel_atomic);
        run_adjointness(cfg_tiled,  "Tiled",  rel_tiled);

        // ====================================================================
        // 2) Consistency test: Atomic vs Tiled scatter
        // ====================================================================
        grid_scattered_atomic = complex_type(0.0, 0.0);
        grid_scattered_tiled  = complex_type(0.0, 0.0);

        bunch.Q_scatter.scatter_kernel(grid_scattered_atomic, bunch.R, kernel, cfg_atomic);
        bunch.Q_scatter.scatter_kernel(grid_scattered_tiled,  bunch.R, kernel, cfg_tiled);

        grid_scattered_atomic.accumulateHalo();
        grid_scattered_tiled.accumulateHalo();

        auto gA_view = grid_scattered_atomic.getView();
        auto gT_view = grid_scattered_tiled.getView();

        std::size_t n_local_i = lDom[0].length();
        std::size_t n_local_j = lDom[1].length();
        std::size_t n_local_k = lDom[2].length();

        // Compute max diff and norm on GPU
        real_type max_diff_scatter_local = 0.0;
        real_type norm_scatter_local     = 0.0;
        const int ng_cmp = nghost;

        Kokkos::parallel_reduce("ScatterCompare",
            mdrange_policy({0, 0, 0},
                          {static_cast<int>(n_local_k),
                           static_cast<int>(n_local_j),
                           static_cast<int>(n_local_i)}),
            KOKKOS_LAMBDA(const int kk, const int jj, const int ii,
                         real_type& max_diff, real_type& norm_sum) {
                complex_type a = gA_view(ii + ng_cmp, jj + ng_cmp, kk + ng_cmp);
                complex_type t = gT_view(ii + ng_cmp, jj + ng_cmp, kk + ng_cmp);

                real_type diff = Kokkos::abs(a - t);
                if (diff > max_diff) max_diff = diff;
                norm_sum += a.real() * a.real() + a.imag() * a.imag();
            },
            Kokkos::Max<real_type>(max_diff_scatter_local),
            Kokkos::Sum<real_type>(norm_scatter_local)
        );

        real_type max_diff_scatter, norm_scatter;
        ippl::Comm->reduce(max_diff_scatter_local, max_diff_scatter, 1,
                           std::greater<real_type>());
        ippl::Comm->reduce(norm_scatter_local, norm_scatter, 1,
                           std::plus<real_type>());

        real_type rel_scatter = 0.0;
        if (rank == 0) {
            rel_scatter =
                max_diff_scatter / std::max<real_type>(std::sqrt(norm_scatter), 1.0);

            std::cout << "Scatter Atomic vs Tiled:\n";
            std::cout << "  max_abs_diff = " << max_diff_scatter << "\n";
            std::cout << "  rel_diff     = " << rel_scatter << "\n";
        }

        // ====================================================================
        // 3) Consistency test: Atomic vs Tiled gather
        // ====================================================================
        bunch.Q_gather       = complex_type(0.0, 0.0);
        bunch.Q_gather_tiled = complex_type(0.0, 0.0);

        bunch.Q_gather.gather(grid_random, bunch.R, kernel, false, cfg_atomic);
        bunch.Q_gather_tiled.gather(grid_random, bunch.R, kernel, false, cfg_tiled);

        auto Qa_view = bunch.Q_gather.getView();
        auto Qt_view = bunch.Q_gather_tiled.getView();

        // Compute max diff and norm on GPU
        real_type max_diff_gather_local = 0.0;
        real_type norm_gather_local     = 0.0;

        Kokkos::parallel_reduce("GatherCompare",
            Kokkos::RangePolicy<ExecSpace>(0, nLoc),
            KOKKOS_LAMBDA(const std::size_t i,
                         real_type& max_diff, real_type& norm_sum) {
                complex_type a = Qa_view(i);
                complex_type t = Qt_view(i);

                real_type diff = Kokkos::abs(a - t);
                if (diff > max_diff) max_diff = diff;
                norm_sum += a.real() * a.real() + a.imag() * a.imag();
            },
            Kokkos::Max<real_type>(max_diff_gather_local),
            Kokkos::Sum<real_type>(norm_gather_local)
        );

        real_type max_diff_gather, norm_gather;
        ippl::Comm->reduce(max_diff_gather_local, max_diff_gather, 1,
                           std::greater<real_type>());
        ippl::Comm->reduce(norm_gather_local, norm_gather, 1,
                           std::plus<real_type>());

        real_type rel_gather = 0.0;
        if (rank == 0) {
            rel_gather =
                max_diff_gather / std::max<real_type>(std::sqrt(norm_gather), 1.0);

            std::cout << "Gather Atomic vs Tiled:\n";
            std::cout << "  max_abs_diff = " << max_diff_gather << "\n";
            std::cout << "  rel_diff     = " << rel_gather << "\n";
        }

        // ====================================================================
        // Final checks / exit code
        // ====================================================================
        const real_type adj_tol    = 1e-12;
        const real_type method_tol = 1e-12;

        if (rank == 0) {
            if (rel_atomic > adj_tol || rel_tiled > adj_tol) {
                std::cerr << "ERROR: adjointness failed (Atomic and/or Tiled)\n";
                exit_code = EXIT_FAILURE;
            }

            if (rel_scatter > method_tol || rel_gather > method_tol) {
                std::cerr << "ERROR: Atomic vs Tiled consistency failed\n";
                exit_code = EXIT_FAILURE;
            }
        }
    }

    ippl::finalize();
    return exit_code;
}