#ifndef IPPL_NUFFT_UTILITIES_H
#define IPPL_NUFFT_UTILITIES_H

/**
 * @file NUFFTUtilities.h
 * @brief NUFFT utility functions adapted from kokkos_nufft
 *
 * Contains:
 * - Type definitions (array, view_nd_t, make_view)
 * - apply_correction: unified correction for Type 1 and Type 2
 * - compute_deconvolution_factors: computes deconvolution factors using Gauss-Legendre quadrature
 */

#include <chrono>
#include <cmath>
#include <limits>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Array.hpp>

#include "ESKernel.h"

namespace ippl {
namespace nufft {

    // ====================================================================
    // Type Definitions
    // ====================================================================

    template<typename T, size_t n>
    using array = Kokkos::Array<T, n>;

    // Multi-dimensional view type helper
    template<class T, int N>
    struct add_ptr_n : add_ptr_n<std::add_pointer_t<T>, N - 1> {};

    template<class T>
    struct add_ptr_n<T, 0> {
        using type = T;
    };

    template<class T, int N>
    using add_ptr_n_t = typename add_ptr_n<T, N>::type;

    template<class T, int Dim, class... Properties>
    using view_nd_t = Kokkos::View<add_ptr_n_t<T, Dim>, Properties...>;

    template<class T, int Dim, class MemorySpace, std::size_t... I>
    static view_nd_t<T, Dim, MemorySpace>
    make_view_impl(const std::string_view label, const array<typename MemorySpace::size_type, Dim> &n,
                   std::index_sequence<I...>) {
        return view_nd_t<T, Dim, MemorySpace>(std::string(label),
                                              static_cast<typename MemorySpace::size_type>(n[I])...);
    }

    template<class T, int Dim, class MemorySpace>
    view_nd_t<T, Dim, MemorySpace>
    make_view(std::string_view label, const array<typename MemorySpace::size_type, Dim> &n) {
        static_assert(Dim > 0, "Dim must be >= 1");
        return make_view_impl<T, Dim, MemorySpace>(label, n, std::make_index_sequence<Dim>{});
    }

    // ====================================================================
    // Quadrature
    // ====================================================================

    /**
     * @brief Computes Gauss-Legendre quadrature nodes and weights.
     */
    template<typename ExecSpace, typename RealType = double>
    void gauss_legendre(int n,
                        Kokkos::View<RealType *, typename ExecSpace::memory_space> &nodes,
                        Kokkos::View<RealType *, typename ExecSpace::memory_space> &weights) {
        constexpr RealType eps = std::numeric_limits<RealType>::epsilon();

        // Create host views to compute values
        auto h_nodes = Kokkos::create_mirror_view(Kokkos::HostSpace(), nodes);
        auto h_weights = Kokkos::create_mirror_view(Kokkos::HostSpace(), weights);

        for (int i = 0; i < n; ++i) {
            RealType x = std::cos(M_PI * (i + 0.75) / (n + 0.5));
            RealType pp, delta;

            do {
                RealType p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < n; ++j) {
                    const RealType p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j + 1.0) * x * p2 - j * p3) / (j + 1.0);
                }
                pp = n * (x * p1 - p2) / (x * x - 1.0);
                delta = p1 / pp;
                x -= delta;
            } while (std::abs(delta) > eps);

            h_nodes(i) = x;
            h_weights(i) = 2.0 / ((1.0 - x * x) * pp * pp);
        }

        // Exploit symmetry
        for (int i = 0; i < n / 2; ++i) {
            const int j = n - 1 - i;
            h_nodes(j) = -h_nodes(i);
            h_weights(j) = h_weights(i);
        }

        // Copy to device
        Kokkos::deep_copy(nodes, h_nodes);
        Kokkos::deep_copy(weights, h_weights);
    }

    /**
     * @brief Computes deconvolution factors for NUFFT.
     */
    template<class ExecSpace, typename RealType = double>
    void compute_deconvolution_factors(
        Kokkos::View<Kokkos::complex<RealType> *, typename ExecSpace::memory_space> factors,
        int64_t n_modes,
        int64_t n_grid,
        const ippl::NUFFT::ESKernel<RealType> &kernel) {
        using complex_type = Kokkos::complex<RealType>;

        // Set up quadrature
        constexpr int q = 100;
        auto nodes = Kokkos::View<RealType *, typename ExecSpace::memory_space>("nodes", q);
        auto weights = Kokkos::View<RealType *, typename ExecSpace::memory_space>("weights", q);

        gauss_legendre<ExecSpace>(q, nodes, weights);

        const RealType alpha = M_PI * kernel.width() / n_grid;
        const RealType beta = kernel.beta();

        const int64_t l_n_modes = n_modes;
        constexpr int l_q = q;

        Kokkos::parallel_for("compute_deconv_factors",
                             Kokkos::RangePolicy<ExecSpace>(0, n_modes),
                             KOKKOS_LAMBDA(const int64_t k) {
                                 int freq = (k < l_n_modes / 2) ? k : k - l_n_modes;
                                 RealType ft = 0.0;

                                 // Compute Fourier transform of kernel via quadrature
                                 for (int i = 0; i < l_q; ++i) {
                                     const RealType x = nodes(i);
                                     const RealType w = weights(i);
                                     const RealType ker = Kokkos::exp(beta * (Kokkos::sqrt(1.0 - x * x) - 1.0));
                                     ft += w * ker * Kokkos::cos(freq * alpha * x);
                                 }

                                 // Store deconvolution factor
                                 factors(k) = complex_type(2.0 / (kernel.width() * ft), 0.0);
                             });
    }

    // ====================================================================
    // Correction (Type 1 and Type 2)
    // ====================================================================

    /**
     * @brief Unified functor for applying correction (Type 1 or Type 2).
     */
    template<int Dim, class ExecSpace, typename RealType = double>
    struct CorrectionFunctor {
        using exec_space = ExecSpace;
        using memory_space = typename exec_space::memory_space;
        using real_type = RealType;
        using complex_type = Kokkos::complex<real_type>;
        using size_type = typename memory_space::size_type;
        using complex_view = Kokkos::View<complex_type *, memory_space>;
        using grid_view_type = view_nd_t<complex_type, Dim, memory_space>;

        grid_view_type input;
        array<complex_view, Dim> factors;
        grid_view_type output;
        array<size_type, Dim> n_modes;
        array<size_type, Dim> n_input;
        array<size_type, Dim> n_output;

        template<typename... Indices>
        KOKKOS_INLINE_FUNCTION void operator()(Indices... indices) const {
            array<int64_t, Dim> k{static_cast<int64_t>(indices)...};

            // Map NUFFT mode indices to FFT grid indices
            array<int64_t, Dim> in_idx, out_idx;
            complex_type factor(1.0, 0.0);

            for (int d = 0; d < Dim; ++d) {
                // Frequency wrapping for FFT layout
                in_idx[d] = (k[d] < n_modes[d] / 2) ? k[d] : n_input[d] - (n_modes[d] - k[d]);
                out_idx[d] = (k[d] < n_modes[d] / 2) ? k[d] : n_output[d] - (n_modes[d] - k[d]);
                factor *= factors[d](k[d]);
            }

            // Apply correction: output = input * factor
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                output(out_idx[Is]...) = input(in_idx[Is]...) * factor;
            }(std::make_index_sequence<Dim>{});
        }
    };

    /**
     * @brief Applies correction (deconvolution or pre-correction).
     *
     * Type 1 (post-FFT):  grid -> output,  n_input = n_output = n_grid
     * Type 2 (pre-FFT):   input -> grid,   n_input = n_modes, n_output = n_grid
     */
    template<class ExecSpace, typename RealType = double, int Dim>
    void apply_correction(
        view_nd_t<Kokkos::complex<RealType>, Dim, typename ExecSpace::memory_space> input,
        array<Kokkos::View<Kokkos::complex<RealType> *, typename ExecSpace::memory_space>, Dim> factors,
        view_nd_t<Kokkos::complex<RealType>, Dim, typename ExecSpace::memory_space> output,
        const array<typename ExecSpace::memory_space::size_type, Dim> &n_modes,
        const array<typename ExecSpace::memory_space::size_type, Dim> &n_input,
        const array<typename ExecSpace::memory_space::size_type, Dim> &n_output,
        RealType &timing) {
        const auto start_time = std::chrono::high_resolution_clock::now();

        // Zero output if we're zero-padding (Type 2: n_output > n_modes)
        bool needs_zero_pad = false;
        for (int d = 0; d < Dim; ++d) {
            if (n_output[d] > n_modes[d]) {
                needs_zero_pad = true;
                break;
            }
        }
        if (needs_zero_pad) {
            Kokkos::deep_copy(output, Kokkos::complex<RealType>(0.0, 0.0));
        }

        // Construct iteration policy over modes
        auto policy = [&] {
            if constexpr (Dim == 1) {
                return Kokkos::RangePolicy<ExecSpace>(0, n_modes[0]);
            } else {
                return [&]<size_t... Is>(std::index_sequence<Is...>) {
                    return Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<Dim> >(
                        {{((void) Is, 0)...}}, {{static_cast<int>(n_modes[Is])...}});
                }(std::make_index_sequence<Dim>{});
            }
        }();

        CorrectionFunctor<Dim, ExecSpace, RealType> functor{
            input, factors, output, n_modes, n_input, n_output
        };

        Kokkos::parallel_for("apply_correction", policy, functor);
        Kokkos::fence();

        const auto end_time = std::chrono::high_resolution_clock::now();
        timing = std::chrono::duration<RealType>(end_time - start_time).count();
    }

} // namespace nufft
} // namespace ippl

#endif // IPPL_NUFFT_UTILITIES_H
