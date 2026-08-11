//
// GaussLegendre1D — Gauss–Legendre (Jacobi α = β = 0) on [-1, 1].
//
#ifndef IPPL_NODES1D_GAUSS_LEGENDRE_1D_H
#define IPPL_NODES1D_GAUSS_LEGENDRE_1D_H

#include <cassert>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Nodes1D/GaussJacobi1D.h"

namespace ippl {
namespace nodes1d {

    template <typename Scalar = double>
    void computeGaussLegendre(std::size_t n, Scalar* nodes, Scalar* weights,
                              std::size_t maxNewtonIterations = 40,
                              std::size_t minNewtonIterations  = 1,
                              InitialGuessType initialGuess = InitialGuessType::Asymptotic,
                              RootFinderMethod method = RootFinderMethod::GolubWelsch) {
        computeGaussJacobi(n, Scalar(0), Scalar(0), nodes, weights, maxNewtonIterations,
                           minNewtonIterations, initialGuess, method);
    }

    template <typename T, unsigned N>
    void computeGaussLegendre(Vector<T, N>& nodes, Vector<T, N>& weights,
                              std::size_t maxNewtonIterations = 40,
                              std::size_t minNewtonIterations  = 1,
                              InitialGuessType initialGuess = InitialGuessType::Asymptotic,
                              RootFinderMethod method = RootFinderMethod::GolubWelsch) {
        computeGaussJacobi(nodes, weights, T(0), T(0), maxNewtonIterations, minNewtonIterations,
                           initialGuess, method);
    }

    template <typename ExecSpace, typename RealType = double>
    void computeGaussLegendre(Kokkos::View<RealType*, typename ExecSpace::memory_space>& nodes,
                              Kokkos::View<RealType*, typename ExecSpace::memory_space>& weights,
                              std::size_t maxNewtonIterations = 40,
                              std::size_t minNewtonIterations  = 1,
                              InitialGuessType initialGuess = InitialGuessType::Asymptotic,
                              RootFinderMethod method = RootFinderMethod::GolubWelsch) {
        const int n = static_cast<int>(nodes.extent(0));
        assert(weights.extent(0) == nodes.extent(0));
        assert(n >= 1);

        auto h_nodes   = Kokkos::create_mirror_view(Kokkos::HostSpace(), nodes);
        auto h_weights = Kokkos::create_mirror_view(Kokkos::HostSpace(), weights);

        std::vector<RealType> nbuf(static_cast<std::size_t>(n));
        std::vector<RealType> wbuf(static_cast<std::size_t>(n));
        computeGaussLegendre(static_cast<std::size_t>(n), nbuf.data(), wbuf.data(),
                             maxNewtonIterations, minNewtonIterations, initialGuess, method);
        for (int i = 0; i < n; ++i) {
            h_nodes(i)   = nbuf[static_cast<std::size_t>(i)];
            h_weights(i) = wbuf[static_cast<std::size_t>(i)];
        }
        Kokkos::deep_copy(nodes, h_nodes);
        Kokkos::deep_copy(weights, h_weights);
    }

}  // namespace nodes1d
}  // namespace ippl

#endif
