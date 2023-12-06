// Class GaussJacobiQuadrature
//   See: https://craffael.github.io/lehrfempp/gauss__quadrature_8cc_source.html

#ifndef IPPL_GAUSSJACOBIQUADRATURE_H
#define IPPL_GAUSSJACOBIQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"

namespace ippl {

    enum InitialGuessType {
        Chebyshev,
        LehrFEM,
    };

    template <typename T, unsigned NumNodes1D, typename ElementType>
    class GaussJacobiQuadrature : public Quadrature<T, NumNodes1D, ElementType> {
    public:
        using scalar_t = long double;  // might be equivalant to double, depending on compiler

        GaussJacobiQuadrature(const ElementType& ref_element, const T& alpha, const T& beta,
                              const std::size_t& max_newton_itersations = 10,
                              const std::size_t& min_newton_iterations  = 1);

        void computeNodesAndWeights() override;

        scalar_t getChebyshevNodes(const std::size_t& i) const;  // TODO maybe move somewhere else?

    private:
        scalar_t getLehrFEMInitialGuess(
            const std::size_t& i, const Vector<scalar_t, NumNodes1D>& integration_nodes) const;

        const T alpha_m;
        const T beta_m;

        const std::size_t max_newton_iterations_m;
        const std::size_t min_newton_iterations_m;
    };

    template <typename T, unsigned NumNodes1D, typename ElementType>
    class GaussLegendreQuadrature : public GaussJacobiQuadrature<T, NumNodes1D, ElementType> {
    public:
        GaussLegendreQuadrature(const ElementType& ref_element,
                                const std::size_t& max_newton_itersations = 10,
                                const std::size_t& min_newton_iterations  = 1)
            : GaussJacobiQuadrature<T, NumNodes1D, ElementType>(
                ref_element, 0.0, 0.0, max_newton_itersations, min_newton_iterations) {}
    };

    /**
     * @brief
     *
     * @tparam T
     * @tparam NumNodes1D
     * @tparam ElementType
     */
    template <typename T, unsigned NumNodes1D, typename ElementType>
    class ChebyshevGaussQuadrature : public GaussJacobiQuadrature<T, NumNodes1D, ElementType> {
    public:
        ChebyshevGaussQuadrature(const ElementType& ref_element,
                                 const std::size_t& max_newton_itersations = 10,
                                 const std::size_t& min_newton_iterations  = 1)
            : GaussJacobiQuadrature<T, NumNodes1D, ElementType>(
                ref_element, -0.5, -0.5, max_newton_itersations, min_newton_iterations) {}
    };

}  // namespace ippl

#include "FEM/Quadrature/GaussJacobiQuadrature.hpp"

#endif