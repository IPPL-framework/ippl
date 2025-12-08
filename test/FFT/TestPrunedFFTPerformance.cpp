#include "Ippl.h"

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "Utility/ParameterList.h"

void benchmarkPrunedCC(int warmup_runs, int benchmark_runs) {
    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;

    std::array<int, dim> pt_full   = {128, 128, 128};
    std::array<int, dim> pt_pruned = {64, 64, 64};

    // Create layouts
    ippl::Index I_full(pt_full[0]);
    ippl::Index J_full(pt_full[1]);
    ippl::Index K_full(pt_full[2]);
    ippl::NDIndex<dim> owned_full(I_full, J_full, K_full);

    ippl::Index I_pruned(pt_pruned[0]);
    ippl::Index J_pruned(pt_pruned[1]);
    ippl::Index K_pruned(pt_pruned[2]);
    ippl::NDIndex<dim> owned_pruned(I_pruned, J_pruned, K_pruned);

    std::array<bool, dim> isParallel;
    isParallel.fill(true);

    ippl::FieldLayout<dim> layout_full(MPI_COMM_WORLD, owned_full, isParallel);
    ippl::FieldLayout<dim> layout_pruned(MPI_COMM_WORLD, owned_pruned, isParallel);

    std::array<double, dim> dx = {
        1.0 / double(pt_full[0]),
        1.0 / double(pt_full[1]),
        1.0 / double(pt_full[2]),
    };
    ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
    ippl::Vector<double, 3> origin = {0, 0, 0};
    Mesh_t mesh_full(owned_full, hx, origin);
    Mesh_t mesh_pruned(owned_pruned, hx, origin);

    typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t> field_type;

    field_type field_input(mesh_full, layout_full);
    field_type field_input_copy(mesh_full, layout_full);
    field_type field_full_result(mesh_full, layout_full);
    field_type field_pruned_result(mesh_pruned, layout_pruned);

    // Setup pruning parameters
    ippl::PruningParams<dim> pruning;
    pruning.n_modes = ippl::Vector<size_t, dim>{static_cast<size_t>(pt_pruned[0]),
                                                static_cast<size_t>(pt_pruned[1]),
                                                static_cast<size_t>(pt_pruned[2])};

    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", true);

    // Create FFTs
    typedef ippl::FFT<ippl::PrunedCCTransform, field_type> PrunedFFT_type;
    typedef ippl::FFT<ippl::CCTransform, field_type> FFT_type;

    auto pruned_fft =
        std::make_unique<PrunedFFT_type>(layout_full, layout_pruned, pruning, fftParams);
    auto regular_fft = std::make_unique<FFT_type>(layout_full, fftParams);

    if (ippl::Comm->rank() == 0) {
        std::cout << "\n=== Benchmarking Pruned C2C FFT ===" << std::endl;
        std::cout << "Full grid: " << pt_full[0] << "x" << pt_full[1] << "x" << pt_full[2]
                  << std::endl;
        std::cout << "Pruned to: " << pt_pruned[0] << "x" << pt_pruned[1] << "x" << pt_pruned[2]
                  << std::endl;
        std::cout << "Warmup runs: " << warmup_runs << std::endl;
        std::cout << "Benchmark runs: " << benchmark_runs << std::endl;
        std::cout << std::endl;
    }

    // Initialize with random data
    const int nghost                           = field_input.getNghost();
    auto& view_full                            = field_input.getView();
    typename field_type::HostMirror field_host = field_input.getHostMirror();

    std::mt19937_64 eng(42 + ippl::Comm->rank());
    std::uniform_real_distribution<double> unif(-1.0, 1.0);

    for (size_t i = nghost; i < view_full.extent(0) - nghost; ++i) {
        for (size_t j = nghost; j < view_full.extent(1) - nghost; ++j) {
            for (size_t k = nghost; k < view_full.extent(2) - nghost; ++k) {
                field_host(i, j, k) = Kokkos::complex<double>(unif(eng), unif(eng));
            }
        }
    }
    Kokkos::deep_copy(field_input.getView(), field_host);

    // Prepare index mapping for manual pruning
    const int N0 = pt_full[0], K0 = pt_pruned[0];
    const int N1 = pt_full[1], K1 = pt_pruned[1];
    const int N2 = pt_full[2], K2 = pt_pruned[2];

    const auto& lDom_pruned     = layout_pruned.getLocalNDIndex();
    const auto& lDom_full       = layout_full.getLocalNDIndex();
    const int nghost_pruned     = field_pruned_result.getNghost();

    const int p0_first = lDom_pruned[0].first();
    const int p1_first = lDom_pruned[1].first();
    const int p2_first = lDom_pruned[2].first();

    const int f0_first = lDom_full[0].first();
    const int f1_first = lDom_full[1].first();
    const int f2_first = lDom_full[2].first();

    // ========== Warmup ==========
    if (ippl::Comm->rank() == 0) {
        std::cout << "Running warmup..." << std::endl;
    }

    for (int i = 0; i < warmup_runs; ++i) {
        // Warmup full FFT + prune
        field_full_result = field_input;
        regular_fft->transform(ippl::FORWARD, field_full_result);
        Kokkos::fence();

        // Warmup pruned FFT
        field_input_copy = field_input;
        pruned_fft->transform(ippl::FORWARD, field_input_copy, field_pruned_result);
        Kokkos::fence();
    }

    MPI_Barrier(ippl::Comm->getCommunicator());

    // ========== Benchmark Full FFT + Manual Prune ==========
    std::vector<double> times_full_prune(benchmark_runs);

    if (ippl::Comm->rank() == 0) {
        std::cout << "Benchmarking full FFT + manual prune..." << std::endl;
    }

    for (int run = 0; run < benchmark_runs; ++run) {
        field_full_result = field_input;
        
        MPI_Barrier(ippl::Comm->getCommunicator());
        auto start = std::chrono::high_resolution_clock::now();

        // Full FFT
        regular_fft->transform(ippl::FORWARD, field_full_result);
        Kokkos::fence();

        // Manual pruning: extract modes from full result to pruned field
        auto &view_full_result = field_full_result.getView();
        auto &view_pruned_out  = field_pruned_result.getView();

        const int ng   = nghost;
        const int ng_p = nghost_pruned;

        using exec_space = typename field_type::execution_space;
        using mdrange_t  = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

        // Just compare without pruning the runtime
        // Kokkos::parallel_for(
        //     "ManualPrune",
        //     mdrange_t({ng_p, ng_p, ng_p},
        //               {view_pruned_out.extent(0) - ng_p, view_pruned_out.extent(1) - ng_p,
        //                view_pruned_out.extent(2) - ng_p}),
        //     KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p) {
        //         int gi_p = li_p - ng_p + p0_first;
        //         int gj_p = lj_p - ng_p + p1_first;
        //         int gk_p = lk_p - ng_p + p2_first;
        //
        //         int gi_f = (gi_p < K0 / 2) ? gi_p : (N0 - K0 + gi_p);
        //         int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
        //         int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);
        //
        //         int li_f = gi_f - f0_first + ng;
        //         int lj_f = gj_f - f1_first + ng;
        //         int lk_f = gk_f - f2_first + ng;
        //
        //         view_pruned_out(li_p, lj_p, lk_p) = view_full_result(li_f, lj_f, lk_f);
        //     });

        Kokkos::fence();
        MPI_Barrier(ippl::Comm->getCommunicator());

        auto end = std::chrono::high_resolution_clock::now();
        times_full_prune[run] =
            std::chrono::duration<double, std::milli>(end - start).count();
    }

    // ========== Benchmark Pruned FFT ==========
    std::vector<double> times_pruned(benchmark_runs);

    if (ippl::Comm->rank() == 0) {
        std::cout << "Benchmarking pruned FFT..." << std::endl;
    }

    for (int run = 0; run < benchmark_runs; ++run) {
        field_input_copy = field_input;

        MPI_Barrier(ippl::Comm->getCommunicator());
        Kokkos::fence();
        auto start = std::chrono::high_resolution_clock::now();

        pruned_fft->transform(ippl::FORWARD, field_input_copy, field_pruned_result);
        Kokkos::fence();

        MPI_Barrier(ippl::Comm->getCommunicator());
        auto end = std::chrono::high_resolution_clock::now();

        times_pruned[run] =
            std::chrono::duration<double, std::milli>(end - start).count();
    }

    // ========== Compute Statistics ==========
    auto compute_stats = [](const std::vector<double>& times) {
        double sum = 0.0, sum_sq = 0.0;
        double min_t = times[0], max_t = times[0];
        for (double t : times) {
            sum += t;
            sum_sq += t * t;
            min_t = std::min(min_t, t);
            max_t = std::max(max_t, t);
        }
        double mean   = sum / times.size();
        double stddev = std::sqrt(sum_sq / times.size() - mean * mean);
        return std::make_tuple(mean, stddev, min_t, max_t);
    };

    auto [mean_full, std_full, min_full, max_full] = compute_stats(times_full_prune);
    auto [mean_pruned, std_pruned, min_pruned, max_pruned] = compute_stats(times_pruned);

    // Reduce across MPI ranks (use max time as the overall time)
    double global_mean_full, global_mean_pruned;
    double global_min_full, global_min_pruned;
    double global_max_full, global_max_pruned;

    MPI_Allreduce(&mean_full, &global_mean_full, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&mean_pruned, &global_mean_pruned, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&min_full, &global_min_full, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&min_pruned, &global_min_pruned, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&max_full, &global_max_full, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&max_pruned, &global_max_pruned, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());

    // ========== Print Results ==========
    if (ippl::Comm->rank() == 0) {
        std::cout << "\n=== Benchmark Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);

        std::cout << "\nFull FFT + Manual Prune:" << std::endl;
        std::cout << "  Mean time:  " << global_mean_full << " ms" << std::endl;
        std::cout << "  Min time:   " << global_min_full << " ms" << std::endl;
        std::cout << "  Max time:   " << global_max_full << " ms" << std::endl;

        std::cout << "\nPruned FFT:" << std::endl;
        std::cout << "  Mean time:  " << global_mean_pruned << " ms" << std::endl;
        std::cout << "  Min time:   " << global_min_pruned << " ms" << std::endl;
        std::cout << "  Max time:   " << global_max_pruned << " ms" << std::endl;

        std::cout << "\nSpeedup (Full+Prune / Pruned):" << std::endl;
        std::cout << "  Mean: " << std::setprecision(2) << global_mean_full / global_mean_pruned
                  << "x" << std::endl;

        std::cout << "\n=== End Benchmark ===" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    int warmup_runs    = 5;
    int benchmark_runs = 20;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) {
            warmup_runs = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            benchmark_runs = std::atoi(argv[++i]);
        }
    }

    benchmarkPrunedCC(warmup_runs, benchmark_runs);

    ippl::finalize();
    return 0;
}