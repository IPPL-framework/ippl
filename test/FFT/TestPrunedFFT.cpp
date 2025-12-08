#include "Ippl.h"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "Utility/ParameterList.h"

// Test pruned C2C FFT by comparing with full FFT
bool testPrunedCC() {
    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;

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

    field_type field_input(mesh_full, layout_full);
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
        std::cout << "\n=== Testing Pruned C2C FFT ===" << std::endl;
        std::cout << "Full grid: " << pt_full[0] << "x" << pt_full[1] << "x" << pt_full[2]
                  << std::endl;
        std::cout << "Pruned to: " << pt_pruned[0] << "x" << pt_pruned[1] << "x" << pt_pruned[2]
                  << std::endl;
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

    // Save input and compute both FFTs
    field_full_result = field_input;

    pruned_fft->transform(ippl::FORWARD, field_input, field_pruned_result);
    regular_fft->transform(ippl::FORWARD, field_full_result);

    // Compare pruned result with corresponding modes from full result
    auto view_pruned      = field_pruned_result.getView();
    auto view_full_result = field_full_result.getView();

    const auto& lDom_pruned = layout_pruned.getLocalNDIndex();
    const auto& lDom_full   = layout_full.getLocalNDIndex();
    const int nghost_pruned = field_pruned_result.getNghost();

    const int N0 = pt_full[0], K0 = pt_pruned[0];
    const int N1 = pt_full[1], K1 = pt_pruned[1];
    const int N2 = pt_full[2], K2 = pt_pruned[2];

    const int p0_first = lDom_pruned[0].first(), p0_last = lDom_pruned[0].last();
    const int p1_first = lDom_pruned[1].first(), p1_last = lDom_pruned[1].last();
    const int p2_first = lDom_pruned[2].first(), p2_last = lDom_pruned[2].last();

    const int f0_first = lDom_full[0].first(), f0_last = lDom_full[0].last();
    const int f1_first = lDom_full[1].first(), f1_last = lDom_full[1].last();
    const int f2_first = lDom_full[2].first(), f2_last = lDom_full[2].last();

    // {
    //     // Copy to host for printing
    //     typename field_type::HostMirror host_pruned = field_pruned_result.getHostMirror();
    //     typename field_type::HostMirror host_full   = field_full_result.getHostMirror();
    //     Kokkos::deep_copy(host_pruned, field_pruned_result.getView());
    //     Kokkos::deep_copy(host_full, field_full_result.getView());
    //
    //     if (ippl::Comm->rank() == 0) {
    //         std::cout << "\n=== Comparing Pruned vs Full (truncated) FFT values ===" << std::endl;
    //         std::cout << std::setw(20) << "Pruned Index" << std::setw(20) << "Full Index"
    //                   << std::setw(30) << "Pruned Value" << std::setw(30) << "Full Value"
    //                   << std::setw(15) << "Error" << std::endl;
    //         std::cout << std::string(115, '-') << std::endl;
    //
    //         for (int i = 0; i < pt_pruned[0]; ++i) {
    //             for (int j = 0; j < pt_pruned[1]; ++j) {
    //                 for (int k = 0; k < pt_pruned[2]; ++k) {
    //                     // Map pruned index to full index
    //                     int fi = (i < K0 / 2) ? i : (N0 - K0 + i);
    //                     int fj = (j < K1 / 2) ? j : (N1 - K1 + j);
    //                     int fk = (k < K2 / 2) ? k : (N2 - K2 + k);
    //
    //                     auto val_p =
    //                         host_pruned(i + nghost_pruned, j + nghost_pruned, k + nghost_pruned);
    //                     auto val_f = host_full(fi + nghost, fj + nghost, fk + nghost);
    //                     double err = Kokkos::abs(val_p - val_f);
    //
    //                     std::cout << std::setw(6) << "(" << i << "," << j << "," << k << ")"
    //                               << std::setw(8) << "(" << fi << "," << fj << "," << fk << ")"
    //                               << std::setw(15) << std::fixed << std::setprecision(6) << "("
    //                               << val_p.real() << ", " << val_p.imag() << ")" << std::setw(15)
    //                               << "(" << val_f.real() << ", " << val_f.imag() << ")"
    //                               << std::setw(15) << std::scientific << err << std::endl;
    //                 }
    //             }
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    const int ng   = nghost;
    const int ng_p = nghost_pruned;

    double max_error = 0.0;
    size_t count     = 0;

    using exec_space = typename field_type::execution_space;
    using mdrange_t  = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

    Kokkos::parallel_reduce(
        "ComparePrunedWithFull",
        mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p,
                                       view_pruned.extent(2) - ng_p}),
        KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, double& local_max) {
            int gi_p = li_p - ng_p + p0_first;
            int gj_p = lj_p - ng_p + p1_first;
            int gk_p = lk_p - ng_p + p2_first;

            int gi_f = (gi_p < K0 / 2) ? gi_p : (N0 - K0 + gi_p);
            int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
            int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);

            if (gi_f < f0_first || gi_f > f0_last || gj_f < f1_first || gj_f > f1_last
                || gk_f < f2_first || gk_f > f2_last) {
                return;
            }

            int li_f = gi_f - f0_first + ng;
            int lj_f = gj_f - f1_first + ng;
            int lk_f = gk_f - f2_first + ng;

            auto val_pruned = view_pruned(li_p, lj_p, lk_p);
            auto val_full   = view_full_result(li_f, lj_f, lk_f);

            double error = Kokkos::abs(val_pruned - val_full);
            if (error > local_max) {
                local_max = error;
            }
        },
        Kokkos::Max<double>(max_error));

    Kokkos::fence();

    Kokkos::parallel_reduce(
        "CountComparisons",
        mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p,
                                       view_pruned.extent(2) - ng_p}),
        KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, size_t& local_count) {
            int gi_p = li_p - ng_p + p0_first;
            int gj_p = lj_p - ng_p + p1_first;
            int gk_p = lk_p - ng_p + p2_first;

            int gi_f = (gi_p < K0 / 2) ? gi_p : (N0 - K0 + gi_p);
            int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
            int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);

            if (gi_f >= f0_first && gi_f <= f0_last && gj_f >= f1_first && gj_f <= f1_last
                && gk_f >= f2_first && gk_f <= f2_last) {
                ++local_count;
            }
        },
        count);

    Kokkos::fence();

    double global_max_error;
    size_t global_count;
    MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                  ippl::Comm->getCommunicator());

    bool passed = (global_max_error < 1e-10);

    if (ippl::Comm->rank() == 0) {
        std::cout << "Compared " << global_count << " modes" << std::endl;
        std::cout << "Max error vs full FFT: " << std::scientific << std::setprecision(6)
                  << global_max_error << std::endl;
        std::cout << "Pruned C2C FFT test: " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    return passed;
}

// Test pruned R2C FFT by comparing with full R2C FFT
bool testPrunedRC() {
    constexpr unsigned int dim = 3;
    using Mesh_t               = ippl::UniformCartesian<double, dim>;
    using Centering_t          = Mesh_t::DefaultCentering;

    std::array<int, dim> pt_real           = {64, 64, 64};
    std::array<int, dim> pt_complex_full   = {33, 64, 64};  // R2C in dimension 0: N/2+1
    std::array<int, dim> pt_complex_pruned = {17, 32, 32};  // Pruned modes

    // Create layouts
    ippl::Index I_real(pt_real[0]);
    ippl::Index J_real(pt_real[1]);
    ippl::Index K_real(pt_real[2]);
    ippl::NDIndex<dim> owned_real(I_real, J_real, K_real);

    ippl::Index I_cfull(pt_complex_full[0]);
    ippl::Index J_cfull(pt_complex_full[1]);
    ippl::Index K_cfull(pt_complex_full[2]);
    ippl::NDIndex<dim> owned_complex_full(I_cfull, J_cfull, K_cfull);

    ippl::Index I_cpruned(pt_complex_pruned[0]);
    ippl::Index J_cpruned(pt_complex_pruned[1]);
    ippl::Index K_cpruned(pt_complex_pruned[2]);
    ippl::NDIndex<dim> owned_complex_pruned(I_cpruned, J_cpruned, K_cpruned);

    std::array<bool, dim> isParallel;
    isParallel.fill(true);

    ippl::FieldLayout<dim> layout_real(MPI_COMM_WORLD, owned_real, isParallel);
    ippl::FieldLayout<dim> layout_complex_full(MPI_COMM_WORLD, owned_complex_full, isParallel);
    ippl::FieldLayout<dim> layout_complex_pruned(MPI_COMM_WORLD, owned_complex_pruned, isParallel);

    std::array<double, dim> dx = {
        1.0 / double(pt_real[0]),
        1.0 / double(pt_real[1]),
        1.0 / double(pt_real[2]),
    };
    ippl::Vector<double, 3> hx     = {dx[0], dx[1], dx[2]};
    ippl::Vector<double, 3> origin = {0, 0, 0};

    Mesh_t mesh_real(owned_real, hx, origin);
    Mesh_t mesh_complex_full(owned_complex_full, hx, origin);
    Mesh_t mesh_complex_pruned(owned_complex_pruned, hx, origin);

    typedef ippl::Field<double, dim, Mesh_t, Centering_t>::uniform_type field_type_real;
    typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t>::uniform_type
        field_type_complex;

    field_type_real field_real_input(mesh_real, layout_real);
    field_type_real field_real_copy(mesh_real, layout_real);
    field_type_complex field_complex_full(mesh_complex_full, layout_complex_full);
    field_type_complex field_complex_pruned(mesh_complex_pruned, layout_complex_pruned);

    // Setup pruning parameters
    ippl::PruningParams<dim> pruning;
    pruning.n_modes = ippl::Vector<size_t, dim>{static_cast<size_t>(pt_complex_pruned[0]),
                                                static_cast<size_t>(pt_complex_pruned[1]),
                                                static_cast<size_t>(pt_complex_pruned[2])};

    ippl::ParameterList fftParams;
    fftParams.add("use_heffte_defaults", true);
    fftParams.add("r2c_direction", 0);

    // Create FFTs
    typedef ippl::FFT<ippl::PrunedRCTransform, field_type_real> PrunedRCFFT_type;
    typedef ippl::FFT<ippl::RCTransform, field_type_real> RCFFT_type;

    auto pruned_fft  = std::make_unique<PrunedRCFFT_type>(layout_real, layout_complex_full,
                                                          layout_complex_pruned, pruning, fftParams);
    auto regular_fft = std::make_unique<RCFFT_type>(layout_real, layout_complex_full, fftParams);

    if (ippl::Comm->rank() == 0) {
        std::cout << "\n=== Testing Pruned R2C FFT ===" << std::endl;
        std::cout << "Real grid: " << pt_real[0] << "x" << pt_real[1] << "x" << pt_real[2]
                  << std::endl;
        std::cout << "Full complex: " << pt_complex_full[0] << "x" << pt_complex_full[1] << "x"
                  << pt_complex_full[2] << std::endl;
        std::cout << "Pruned complex: " << pt_complex_pruned[0] << "x" << pt_complex_pruned[1]
                  << "x" << pt_complex_pruned[2] << std::endl;
    }

    // Initialize with random data
    const int nghost                                = field_real_input.getNghost();
    auto& view_real                                 = field_real_input.getView();
    typename field_type_real::HostMirror field_host = field_real_input.getHostMirror();

    std::mt19937_64 eng(42 + ippl::Comm->rank());
    std::uniform_real_distribution<double> unif(-1.0, 1.0);

    for (size_t i = nghost; i < view_real.extent(0) - nghost; ++i) {
        for (size_t j = nghost; j < view_real.extent(1) - nghost; ++j) {
            for (size_t k = nghost; k < view_real.extent(2) - nghost; ++k) {
                field_host(i, j, k) = unif(eng);
            }
        }
    }
    Kokkos::deep_copy(field_real_input.getView(), field_host);

    // Save input and compute both FFTs
    field_real_copy = field_real_input;

    pruned_fft->transform(ippl::FORWARD, field_real_input, field_complex_pruned);
    regular_fft->transform(ippl::FORWARD, field_real_copy, field_complex_full);

    // Compare pruned result with corresponding modes from full result
    // R2C: dimension 0 has only positive freqs [0, N/2], direct mapping
    //      dimensions 1,2 have [0, ..., N/2-1, -N/2, ..., -1], need wrapping
    auto view_pruned = field_complex_pruned.getView();
    auto view_full   = field_complex_full.getView();

    const auto& lDom_pruned = layout_complex_pruned.getLocalNDIndex();
    const auto& lDom_full   = layout_complex_full.getLocalNDIndex();
    const int nghost_pruned = field_complex_pruned.getNghost();
    const int nghost_full   = field_complex_full.getNghost();

    // Full complex dimensions (N/2+1, N, N)
    const int N1 = pt_real[1], K1 = pt_complex_pruned[1];
    const int N2 = pt_real[2], K2 = pt_complex_pruned[2];

    const int p0_first = lDom_pruned[0].first();
    const int p1_first = lDom_pruned[1].first();
    const int p2_first = lDom_pruned[2].first();

    const int f0_first = lDom_full[0].first(), f0_last = lDom_full[0].last();
    const int f1_first = lDom_full[1].first(), f1_last = lDom_full[1].last();
    const int f2_first = lDom_full[2].first(), f2_last = lDom_full[2].last();

    const int ng_p = nghost_pruned;
    const int ng_f = nghost_full;

    double max_error = 0.0;
    size_t count     = 0;

    using exec_space = typename field_type_complex::execution_space;
    using mdrange_t  = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

    Kokkos::parallel_reduce(
        "ComparePrunedWithFullRC",
        mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p,
                                       view_pruned.extent(2) - ng_p}),
        KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, double& local_max) {
            int gi_p = li_p - ng_p + p0_first;
            int gj_p = lj_p - ng_p + p1_first;
            int gk_p = lk_p - ng_p + p2_first;

            // Dimension 0: direct mapping (R2C only has positive freqs)
            int gi_f = gi_p;
            // Dimensions 1,2: wrap negative frequencies
            int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
            int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);

            if (gi_f < f0_first || gi_f > f0_last || gj_f < f1_first || gj_f > f1_last
                || gk_f < f2_first || gk_f > f2_last) {
                return;
            }

            int li_f = gi_f - f0_first + ng_f;
            int lj_f = gj_f - f1_first + ng_f;
            int lk_f = gk_f - f2_first + ng_f;

            auto val_pruned = view_pruned(li_p, lj_p, lk_p);
            auto val_full   = view_full(li_f, lj_f, lk_f);

            double error = Kokkos::abs(val_pruned - val_full);
            if (error > local_max) {
                local_max = error;
            }
        },
        Kokkos::Max<double>(max_error));

    Kokkos::fence();

    Kokkos::parallel_reduce(
        "CountComparisonsRC",
        mdrange_t({ng_p, ng_p, ng_p}, {view_pruned.extent(0) - ng_p, view_pruned.extent(1) - ng_p,
                                       view_pruned.extent(2) - ng_p}),
        KOKKOS_LAMBDA(const int li_p, const int lj_p, const int lk_p, size_t& local_count) {
            int gi_p = li_p - ng_p + p0_first;
            int gj_p = lj_p - ng_p + p1_first;
            int gk_p = lk_p - ng_p + p2_first;

            int gi_f = gi_p;
            int gj_f = (gj_p < K1 / 2) ? gj_p : (N1 - K1 + gj_p);
            int gk_f = (gk_p < K2 / 2) ? gk_p : (N2 - K2 + gk_p);

            if (gi_f >= f0_first && gi_f <= f0_last && gj_f >= f1_first && gj_f <= f1_last
                && gk_f >= f2_first && gk_f <= f2_last) {
                ++local_count;
            }
        },
        count);

    Kokkos::fence();

    double global_max_error;
    size_t global_count;
    MPI_Allreduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX,
                  ippl::Comm->getCommunicator());
    MPI_Allreduce(&count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                  ippl::Comm->getCommunicator());

    bool passed = (global_max_error < 1e-10);

    if (ippl::Comm->rank() == 0) {
        std::cout << "Compared " << global_count << " modes" << std::endl;
        std::cout << "Max error vs full R2C FFT: " << std::scientific << std::setprecision(6)
                  << global_max_error << std::endl;
        std::cout << "Pruned R2C FFT test: " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    return passed;
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    bool cc_passed = testPrunedCC();
    bool rc_passed = testPrunedRC();

    if (ippl::Comm->rank() == 0) {
        std::cout << "\n=== Overall Results ===" << std::endl;
        std::cout << "Pruned C2C: " << (cc_passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Pruned R2C: " << (rc_passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "All tests: " << ((cc_passed && rc_passed) ? "PASSED" : "FAILED") << std::endl;
    }

    ippl::finalize();
    return (cc_passed && rc_passed) ? 0 : 1;
}