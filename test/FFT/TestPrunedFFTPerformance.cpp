#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "Utility/ParameterList.h"

struct BenchmarkResult {
    double mean_full;
    double min_full;
    double max_full;
    double mean_pruned;
    double min_pruned;
    double max_pruned;
    double speedup;
    size_t memory_full_fft_bytes;
    size_t memory_pruned_fft_bytes;
};

struct MemoryInfo {
    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes;
};

MemoryInfo getCudaMemoryInfo() {
    MemoryInfo info = {0, 0, 0};
#ifdef KOKKOS_ENABLE_CUDA
    cudaMemGetInfo(&info.free_bytes, &info.total_bytes);
    info.used_bytes = info.total_bytes - info.free_bytes;
#endif
    return info;
}

void printMemoryUsage(const std::string& label) {
    if (ippl::Comm->rank() == 0) {
#ifdef KOKKOS_ENABLE_CUDA
        MemoryInfo info = getCudaMemoryInfo();
        std::cout << "[Memory] " << label << ": "
                  << "Used: " << (info.used_bytes / (1024.0 * 1024.0)) << " MB, "
                  << "Free: " << (info.free_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
#endif
    }
}

auto compute_stats(const std::vector<double>& times) {
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
}

BenchmarkResult benchmarkForwardFFT(int warmup_runs, int benchmark_runs) {
    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;
    //
    // std::array<int, dim> pt_full   = {1024, 1024, 512};
    // std::array<int, dim> pt_pruned = {512, 512, 256};


    std::array<int, dim> pt_full   = {64, 128, 32};
    std::array<int, dim> pt_pruned = {32, 64, 16};

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

    if (ippl::Comm->rank() == 0) {
        std::cout << "\n=== Benchmarking Forward FFT ===" << std::endl;
        std::cout << "Full grid: " << pt_full[0] << "x" << pt_full[1] << "x" << pt_full[2]
                  << std::endl;
        std::cout << "Pruned to: " << pt_pruned[0] << "x" << pt_pruned[1] << "x" << pt_pruned[2]
                  << std::endl;
        std::cout << "Warmup runs: " << warmup_runs << std::endl;
        std::cout << "Benchmark runs: " << benchmark_runs << std::endl;
    }

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("Before field allocation");

    // Minimal fields: one input field, one output field (reused for both FFT types)
    field_type field_input(mesh_full, layout_full);
    field_type field_output_full(mesh_full, layout_full);
    field_type field_output_pruned(mesh_pruned, layout_pruned);

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("After field allocation");

    // Initialize with random data directly on GPU
    using exec_space = typename field_type::execution_space;
    using RandPool   = Kokkos::Random_XorShift64_Pool<exec_space>;

    const int nghost    = field_input.getNghost();
    auto view_input     = field_input.getView();
    RandPool rand_pool(42 + ippl::Comm->rank());

    Kokkos::parallel_for(
        "InitRandomData",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            {nghost, nghost, nghost}, {view_input.extent(0) - nghost, view_input.extent(1) - nghost,
                                       view_input.extent(2) - nghost}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            auto rand_gen       = rand_pool.get_state();
            double real_part    = rand_gen.drand(-1.0, 1.0);
            double imag_part    = rand_gen.drand(-1.0, 1.0);
            view_input(i, j, k) = Kokkos::complex<double>(real_part, imag_part);
            rand_pool.free_state(rand_gen);
        });
    Kokkos::fence();

    // Setup pruning parameters
    ippl::PruningParams<dim> pruning;
    pruning.n_modes = ippl::Vector<size_t, dim>{static_cast<size_t>(pt_pruned[0]),
                                                static_cast<size_t>(pt_pruned[1]),
                                                static_cast<size_t>(pt_pruned[2])};

    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", false);
    fftParams.add("use_pencils", true);
    fftParams.add("use_reorder", false);
    fftParams.add("use_gpu_aware", true);
    fftParams.add("comm", 2);

    typedef ippl::FFT<ippl::PrunedCCTransform, field_type> PrunedFFT_type;
    typedef ippl::FFT<ippl::CCTransform, field_type> FFT_type;

    size_t memory_full_fft    = 0;
    size_t memory_pruned_fft  = 0;
    std::vector<double> times_fwd_full(benchmark_runs);
    std::vector<double> times_fwd_pruned(benchmark_runs);

    // ========== Benchmark Full FFT ==========
    {
        Kokkos::fence();
        MPI_Barrier(ippl::Comm->getCommunicator());
        MemoryInfo mem_before = getCudaMemoryInfo();
        printMemoryUsage("Before regular FFT allocation");

        auto regular_fft = std::make_unique<FFT_type>(layout_full, fftParams);

        Kokkos::fence();
        MPI_Barrier(ippl::Comm->getCommunicator());
        MemoryInfo mem_after = getCudaMemoryInfo();
        printMemoryUsage("After regular FFT allocation");

        memory_full_fft = (mem_after.used_bytes > mem_before.used_bytes)
                              ? (mem_after.used_bytes - mem_before.used_bytes)
                              : 0;

        if (ippl::Comm->rank() == 0) {
            std::cout << "[Memory] Regular FFT plan size: " << (memory_full_fft / (1024.0 * 1024.0))
                      << " MB" << std::endl;
        }

        // Warmup
        if (ippl::Comm->rank() == 0) {
            std::cout << "Running warmup for full FFT..." << std::endl;
        }

        for (int i = 0; i < warmup_runs; ++i) {
            field_output_full = field_input;
            regular_fft->transform(ippl::FORWARD, field_output_full);
            Kokkos::fence();
        }

        MPI_Barrier(ippl::Comm->getCommunicator());

        // Benchmark
        if (ippl::Comm->rank() == 0) {
            std::cout << "Benchmarking forward full FFT..." << std::endl;
        }

        for (int run = 0; run < benchmark_runs; ++run) {
            field_output_full = field_input;

            MPI_Barrier(ippl::Comm->getCommunicator());
            Kokkos::fence();
            auto start = std::chrono::high_resolution_clock::now();

            regular_fft->transform(ippl::FORWARD, field_output_full);
            Kokkos::fence();

            MPI_Barrier(ippl::Comm->getCommunicator());
            auto end = std::chrono::high_resolution_clock::now();

            times_fwd_full[run] = std::chrono::duration<double, std::milli>(end - start).count();
        }

        // FFT destroyed here, memory released
    }

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("After regular FFT destroyed");

    // ========== Benchmark Pruned FFT ==========
    {
        Kokkos::fence();
        MPI_Barrier(ippl::Comm->getCommunicator());
        MemoryInfo mem_before = getCudaMemoryInfo();
        printMemoryUsage("Before pruned FFT allocation");

        auto pruned_fft =
            std::make_unique<PrunedFFT_type>(layout_full, layout_pruned, pruning, fftParams);

        Kokkos::fence();
        MPI_Barrier(ippl::Comm->getCommunicator());
        MemoryInfo mem_after = getCudaMemoryInfo();
        printMemoryUsage("After pruned FFT allocation");

        memory_pruned_fft = (mem_after.used_bytes > mem_before.used_bytes)
                                ? (mem_after.used_bytes - mem_before.used_bytes)
                                : 0;

        if (ippl::Comm->rank() == 0) {
            std::cout << "[Memory] Pruned FFT plan size: " << (memory_pruned_fft / (1024.0 * 1024.0))
                      << " MB" << std::endl;
        }

        // Warmup
        if (ippl::Comm->rank() == 0) {
            std::cout << "Running warmup for pruned FFT..." << std::endl;
        }

        for (int i = 0; i < warmup_runs; ++i) {
            field_output_full = field_input;  // Reuse as temp input (will be overwritten)
            pruned_fft->transform(ippl::FORWARD, field_output_full, field_output_pruned);
            Kokkos::fence();
        }

        MPI_Barrier(ippl::Comm->getCommunicator());

        // Benchmark
        if (ippl::Comm->rank() == 0) {
            std::cout << "Benchmarking forward pruned FFT..." << std::endl;
        }

        for (int run = 0; run < benchmark_runs; ++run) {
            field_output_full = field_input;  // Reuse as temp input

            MPI_Barrier(ippl::Comm->getCommunicator());
            Kokkos::fence();
            auto start = std::chrono::high_resolution_clock::now();

            pruned_fft->transform(ippl::FORWARD, field_output_full, field_output_pruned);
            Kokkos::fence();

            MPI_Barrier(ippl::Comm->getCommunicator());
            auto end = std::chrono::high_resolution_clock::now();

            times_fwd_pruned[run] = std::chrono::duration<double, std::milli>(end - start).count();
        }

        // FFT destroyed here
    }

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("After pruned FFT destroyed");

    // ========== Compute Statistics ==========
    auto [mean_fwd_full, std_fwd_full, min_fwd_full, max_fwd_full] = compute_stats(times_fwd_full);
    auto [mean_fwd_pruned, std_fwd_pruned, min_fwd_pruned, max_fwd_pruned] =
        compute_stats(times_fwd_pruned);

    double global_mean_fwd_full, global_mean_fwd_pruned;
    double global_min_fwd_full, global_min_fwd_pruned;
    double global_max_fwd_full, global_max_fwd_pruned;

    MPI_Allreduce(&mean_fwd_full, &global_mean_fwd_full, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&mean_fwd_pruned, &global_mean_fwd_pruned, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&min_fwd_full, &global_min_fwd_full, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&min_fwd_pruned, &global_min_fwd_pruned, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&max_fwd_full, &global_max_fwd_full, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&max_fwd_pruned, &global_max_fwd_pruned, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());

    // Get max memory usage across ranks
    size_t global_memory_full_fft, global_memory_pruned_fft;
    MPI_Allreduce(&memory_full_fft, &global_memory_full_fft, 1, MPI_UNSIGNED_LONG, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&memory_pruned_fft, &global_memory_pruned_fft, 1, MPI_UNSIGNED_LONG, MPI_MAX,
                  ippl::Comm->getCommunicator());

    // ========== Print Results ==========
    if (ippl::Comm->rank() == 0) {
        std::cout << "\n--- FORWARD TRANSFORM RESULTS ---" << std::endl;
        std::cout << std::fixed << std::setprecision(3);

        std::cout << "\nFull FFT:" << std::endl;
        std::cout << "  Mean time:  " << global_mean_fwd_full << " ms" << std::endl;
        std::cout << "  Min time:   " << global_min_fwd_full << " ms" << std::endl;
        std::cout << "  Max time:   " << global_max_fwd_full << " ms" << std::endl;
        std::cout << "  FFT Memory: " << (global_memory_full_fft / (1024.0 * 1024.0)) << " MB"
                  << std::endl;

        std::cout << "\nPruned FFT:" << std::endl;
        std::cout << "  Mean time:  " << global_mean_fwd_pruned << " ms" << std::endl;
        std::cout << "  Min time:   " << global_min_fwd_pruned << " ms" << std::endl;
        std::cout << "  Max time:   " << global_max_fwd_pruned << " ms" << std::endl;
        std::cout << "  FFT Memory: " << (global_memory_pruned_fft / (1024.0 * 1024.0)) << " MB"
                  << std::endl;

        std::cout << "\nForward Speedup (Full / Pruned): " << std::setprecision(2)
                  << global_mean_fwd_full / global_mean_fwd_pruned << "x" << std::endl;
    }

    return BenchmarkResult{global_mean_fwd_full,
                           global_min_fwd_full,
                           global_max_fwd_full,
                           global_mean_fwd_pruned,
                           global_min_fwd_pruned,
                           global_max_fwd_pruned,
                           global_mean_fwd_full / global_mean_fwd_pruned,
                           global_memory_full_fft,
                           global_memory_pruned_fft};
}

BenchmarkResult benchmarkBackwardFFT(int warmup_runs, int benchmark_runs) {
    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;

    // std::array<int, dim> pt_full   = {1024, 1024, 512};
    // std::array<int, dim> pt_pruned = {512, 512, 256};
    //
    std::array<int, dim> pt_full   = {64, 128, 32};
    std::array<int, dim> pt_pruned = {32, 64, 16};

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

    if (ippl::Comm->rank() == 0) {
        std::cout << "\n=== Benchmarking Backward FFT ===" << std::endl;
        std::cout << "Full grid: " << pt_full[0] << "x" << pt_full[1] << "x" << pt_full[2]
                  << std::endl;
        std::cout << "Pruned to: " << pt_pruned[0] << "x" << pt_pruned[1] << "x" << pt_pruned[2]
                  << std::endl;
        std::cout << "Warmup runs: " << warmup_runs << std::endl;
        std::cout << "Benchmark runs: " << benchmark_runs << std::endl;
    }

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("Before field allocation");

    // Minimal fields for backward: pruned input, full output
    field_type field_freq_pruned(mesh_pruned, layout_pruned);
    field_type field_freq_full(mesh_full, layout_full);  // For reference full IFFT
    field_type field_output(mesh_full, layout_full);     // Shared output

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("After field allocation");

    using exec_space = typename field_type::execution_space;
    using RandPool   = Kokkos::Random_XorShift64_Pool<exec_space>;

    const int nghost        = field_output.getNghost();
    const int nghost_pruned = field_freq_pruned.getNghost();

    // Initialize pruned frequency input with random data
    auto view_freq_pruned = field_freq_pruned.getView();
    RandPool rand_pool(123 + ippl::Comm->rank());

    Kokkos::parallel_for(
        "InitRandomFreqData",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(
            {nghost_pruned, nghost_pruned, nghost_pruned},
            {view_freq_pruned.extent(0) - nghost_pruned, view_freq_pruned.extent(1) - nghost_pruned,
             view_freq_pruned.extent(2) - nghost_pruned}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            auto rand_gen             = rand_pool.get_state();
            double real_part          = rand_gen.drand(-1.0, 1.0);
            double imag_part          = rand_gen.drand(-1.0, 1.0);
            view_freq_pruned(i, j, k) = Kokkos::complex<double>(real_part, imag_part);
            rand_pool.free_state(rand_gen);
        });
    Kokkos::fence();

    // Zero-pad pruned frequency to full frequency for reference backward transform
    const int N0 = pt_full[0], K0 = pt_pruned[0];
    const int N1 = pt_full[1], K1 = pt_pruned[1];
    const int N2 = pt_full[2], K2 = pt_pruned[2];

    const auto& lDom_pruned = layout_pruned.getLocalNDIndex();
    const auto& lDom_full   = layout_full.getLocalNDIndex();

    const int p0_first = lDom_pruned[0].first();
    const int p1_first = lDom_pruned[1].first();
    const int p2_first = lDom_pruned[2].first();

    const int f0_first = lDom_full[0].first(), f0_last = lDom_full[0].last();
    const int f1_first = lDom_full[1].first(), f1_last = lDom_full[1].last();
    const int f2_first = lDom_full[2].first(), f2_last = lDom_full[2].last();

    auto view_freq_full = field_freq_full.getView();
    Kokkos::deep_copy(view_freq_full, Kokkos::complex<double>(0.0, 0.0));

    using mdrange_t = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

    Kokkos::parallel_for(
        "ZeroPadFrequency",
        mdrange_t(
            {nghost_pruned, nghost_pruned, nghost_pruned},
            {view_freq_pruned.extent(0) - nghost_pruned, view_freq_pruned.extent(1) - nghost_pruned,
             view_freq_pruned.extent(2) - nghost_pruned}),
        KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p) {
            int gi_p = li_p - nghost_pruned + p0_first;
            int gj_p = lj_p - nghost_pruned + p1_first;
            int gk_p = lk_p - nghost_pruned + p2_first;

            int gi_f = (gi_p < K0 / 2) ? gi_p : (N0 - K0 + gi_p);
            int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
            int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);

            if (gi_f >= f0_first && gi_f <= f0_last && gj_f >= f1_first && gj_f <= f1_last
                && gk_f >= f2_first && gk_f <= f2_last) {
                int li_f = gi_f - f0_first + nghost;
                int lj_f = gj_f - f1_first + nghost;
                int lk_f = gk_f - f2_first + nghost;

                view_freq_full(li_f, lj_f, lk_f) = view_freq_pruned(li_p, lj_p, lk_p);
            }
        });
    Kokkos::fence();

    // Store original data for reuse (since transforms are in-place for full FFT)
    field_type field_freq_full_orig(mesh_full, layout_full);
    field_type field_freq_pruned_orig(mesh_pruned, layout_pruned);
    field_freq_full_orig   = field_freq_full;
    field_freq_pruned_orig = field_freq_pruned;

    ippl::PruningParams<dim> pruning;
    pruning.n_modes = ippl::Vector<size_t, dim>{static_cast<size_t>(pt_pruned[0]),
                                                static_cast<size_t>(pt_pruned[1]),
                                                static_cast<size_t>(pt_pruned[2])};

    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", false);
    fftParams.add("use_pencils", true);
    fftParams.add("use_reorder", false);
    fftParams.add("use_gpu_aware", true);
    fftParams.add("comm", 2);

    typedef ippl::FFT<ippl::PrunedCCTransform, field_type> PrunedFFT_type;
    typedef ippl::FFT<ippl::CCTransform, field_type> FFT_type;

    size_t memory_full_fft   = 0;
    size_t memory_pruned_fft = 0;
    std::vector<double> times_bwd_full(benchmark_runs);
    std::vector<double> times_bwd_pruned(benchmark_runs);

    // ========== Benchmark Full IFFT ==========
    {
        Kokkos::fence();
        MPI_Barrier(ippl::Comm->getCommunicator());
        MemoryInfo mem_before = getCudaMemoryInfo();
        printMemoryUsage("Before regular FFT allocation");

        auto regular_fft = std::make_unique<FFT_type>(layout_full, fftParams);

        Kokkos::fence();
        MPI_Barrier(ippl::Comm->getCommunicator());
        MemoryInfo mem_after = getCudaMemoryInfo();
        printMemoryUsage("After regular FFT allocation");

        memory_full_fft = (mem_after.used_bytes > mem_before.used_bytes)
                              ? (mem_after.used_bytes - mem_before.used_bytes)
                              : 0;

        if (ippl::Comm->rank() == 0) {
            std::cout << "[Memory] Regular FFT plan size: " << (memory_full_fft / (1024.0 * 1024.0))
                      << " MB" << std::endl;
        }

        // Warmup
        if (ippl::Comm->rank() == 0) {
            std::cout << "Running warmup for full IFFT..." << std::endl;
        }

        for (int i = 0; i < warmup_runs; ++i) {
            field_freq_full = field_freq_full_orig;
            regular_fft->transform(ippl::BACKWARD, field_freq_full);
            Kokkos::fence();
        }

        MPI_Barrier(ippl::Comm->getCommunicator());

        // Benchmark
        if (ippl::Comm->rank() == 0) {
            std::cout << "Benchmarking backward full IFFT..." << std::endl;
        }

        for (int run = 0; run < benchmark_runs; ++run) {
            field_freq_full = field_freq_full_orig;

            MPI_Barrier(ippl::Comm->getCommunicator());
            Kokkos::fence();
            auto start = std::chrono::high_resolution_clock::now();

            regular_fft->transform(ippl::BACKWARD, field_freq_full);
            Kokkos::fence();

            MPI_Barrier(ippl::Comm->getCommunicator());
            auto end = std::chrono::high_resolution_clock::now();

            times_bwd_full[run] = std::chrono::duration<double, std::milli>(end - start).count();
        }
    }

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("After regular FFT destroyed");

    // ========== Benchmark Pruned IFFT ==========
    {
        Kokkos::fence();
        MPI_Barrier(ippl::Comm->getCommunicator());
        MemoryInfo mem_before = getCudaMemoryInfo();
        printMemoryUsage("Before pruned FFT allocation");

        auto pruned_fft =
            std::make_unique<PrunedFFT_type>(layout_full, layout_pruned, pruning, fftParams);

        Kokkos::fence();
        MPI_Barrier(ippl::Comm->getCommunicator());
        MemoryInfo mem_after = getCudaMemoryInfo();
        printMemoryUsage("After pruned FFT allocation");

        memory_pruned_fft = (mem_after.used_bytes > mem_before.used_bytes)
                                ? (mem_after.used_bytes - mem_before.used_bytes)
                                : 0;

        if (ippl::Comm->rank() == 0) {
            std::cout << "[Memory] Pruned FFT plan size: " << (memory_pruned_fft / (1024.0 * 1024.0))
                      << " MB" << std::endl;
        }

        // Warmup
        if (ippl::Comm->rank() == 0) {
            std::cout << "Running warmup for pruned IFFT..." << std::endl;
        }

        for (int i = 0; i < warmup_runs; ++i) {
            field_freq_pruned = field_freq_pruned_orig;
            pruned_fft->transform(ippl::BACKWARD, field_freq_pruned, field_output);
            Kokkos::fence();
        }

        MPI_Barrier(ippl::Comm->getCommunicator());

        // Benchmark
        if (ippl::Comm->rank() == 0) {
            std::cout << "Benchmarking backward pruned IFFT..." << std::endl;
        }

        for (int run = 0; run < benchmark_runs; ++run) {
            field_freq_pruned = field_freq_pruned_orig;

            MPI_Barrier(ippl::Comm->getCommunicator());
            Kokkos::fence();
            auto start = std::chrono::high_resolution_clock::now();

            pruned_fft->transform(ippl::BACKWARD, field_freq_pruned, field_output);
            Kokkos::fence();

            MPI_Barrier(ippl::Comm->getCommunicator());
            auto end = std::chrono::high_resolution_clock::now();

            times_bwd_pruned[run] = std::chrono::duration<double, std::milli>(end - start).count();
        }
    }

    Kokkos::fence();
    MPI_Barrier(ippl::Comm->getCommunicator());
    printMemoryUsage("After pruned FFT destroyed");

    // ========== Compute Statistics ==========
    auto [mean_bwd_full, std_bwd_full, min_bwd_full, max_bwd_full] = compute_stats(times_bwd_full);
    auto [mean_bwd_pruned, std_bwd_pruned, min_bwd_pruned, max_bwd_pruned] =
        compute_stats(times_bwd_pruned);

    double global_mean_bwd_full, global_mean_bwd_pruned;
    double global_min_bwd_full, global_min_bwd_pruned;
    double global_max_bwd_full, global_max_bwd_pruned;

    MPI_Allreduce(&mean_bwd_full, &global_mean_bwd_full, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&mean_bwd_pruned, &global_mean_bwd_pruned, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&min_bwd_full, &global_min_bwd_full, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&min_bwd_pruned, &global_min_bwd_pruned, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&max_bwd_full, &global_max_bwd_full, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&max_bwd_pruned, &global_max_bwd_pruned, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());

    size_t global_memory_full_fft, global_memory_pruned_fft;
    MPI_Allreduce(&memory_full_fft, &global_memory_full_fft, 1, MPI_UNSIGNED_LONG, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&memory_pruned_fft, &global_memory_pruned_fft, 1, MPI_UNSIGNED_LONG, MPI_MAX,
                  ippl::Comm->getCommunicator());

    // ========== Print Results ==========
    if (ippl::Comm->rank() == 0) {
        std::cout << "\n--- BACKWARD TRANSFORM RESULTS ---" << std::endl;
        std::cout << std::fixed << std::setprecision(3);

        std::cout << "\nFull IFFT:" << std::endl;
        std::cout << "  Mean time:  " << global_mean_bwd_full << " ms" << std::endl;
        std::cout << "  Min time:   " << global_min_bwd_full << " ms" << std::endl;
        std::cout << "  Max time:   " << global_max_bwd_full << " ms" << std::endl;
        std::cout << "  FFT Memory: " << (global_memory_full_fft / (1024.0 * 1024.0)) << " MB"
                  << std::endl;

        std::cout << "\nPruned IFFT:" << std::endl;
        std::cout << "  Mean time:  " << global_mean_bwd_pruned << " ms" << std::endl;
        std::cout << "  Min time:   " << global_min_bwd_pruned << " ms" << std::endl;
        std::cout << "  Max time:   " << global_max_bwd_pruned << " ms" << std::endl;
        std::cout << "  FFT Memory: " << (global_memory_pruned_fft / (1024.0 * 1024.0)) << " MB"
                  << std::endl;

        std::cout << "\nBackward Speedup (Full / Pruned): " << std::setprecision(2)
                  << global_mean_bwd_full / global_mean_bwd_pruned << "x" << std::endl;
    }

    return BenchmarkResult{global_mean_bwd_full,
                           global_min_bwd_full,
                           global_max_bwd_full,
                           global_mean_bwd_pruned,
                           global_min_bwd_pruned,
                           global_max_bwd_pruned,
                           global_mean_bwd_full / global_mean_bwd_pruned,
                           global_memory_full_fft,
                           global_memory_pruned_fft};
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    int warmup_runs    = 5;
    int benchmark_runs = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) {
            warmup_runs = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            benchmark_runs = std::atoi(argv[++i]);
        }
    }

    if (ippl::Comm->rank() == 0) {
        printMemoryUsage("Initial state");
    }

    BenchmarkResult fwd_result = benchmarkForwardFFT(warmup_runs, benchmark_runs);
    BenchmarkResult bwd_result = benchmarkBackwardFFT(warmup_runs, benchmark_runs);

    if (ippl::Comm->rank() == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "=== COMBINED RESULTS ===" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::fixed << std::setprecision(3);

        std::cout << "\n--- ROUND-TRIP (Forward + Backward) ---" << std::endl;
        double total_full   = fwd_result.mean_full + bwd_result.mean_full;
        double total_pruned = fwd_result.mean_pruned + bwd_result.mean_pruned;
        std::cout << "Full (FFT + IFFT):   " << total_full << " ms" << std::endl;
        std::cout << "Pruned (FFT + IFFT): " << total_pruned << " ms" << std::endl;
        std::cout << "Round-trip Speedup:  " << std::setprecision(2) << total_full / total_pruned
                  << "x" << std::endl;

        std::cout << "\n--- MEMORY SUMMARY ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Forward Full FFT:   " << (fwd_result.memory_full_fft_bytes / (1024.0 * 1024.0))
                  << " MB" << std::endl;
        std::cout << "Forward Pruned FFT: "
                  << (fwd_result.memory_pruned_fft_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Backward Full FFT:  " << (bwd_result.memory_full_fft_bytes / (1024.0 * 1024.0))
                  << " MB" << std::endl;
        std::cout << "Backward Pruned FFT: "
                  << (bwd_result.memory_pruned_fft_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

        std::cout << "\n=== End Benchmark ===" << std::endl;
    }

    ippl::finalize();
    return 0;
}