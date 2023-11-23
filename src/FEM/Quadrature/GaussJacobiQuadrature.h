// Class GaussJacobiQuadrature
//   See: https://craffael.github.io/lehrfempp/gauss__quadrature_8cc_source.html

#ifndef IPPL_GAUSSJACOBIQUADRATURE_H
#define IPPL_GAUSSJACOBIQUADRATURE_H

#include "FEM/Quadrature/Quadrature.h"

namespace ippl {

    template <typename T, unsigned NumNodes1D, typename ElementType>
    class GaussJacobiQuadrature : public Quadrature<T, NumNodes1D, ElementType> {
    public:
        GaussJacobiQuadrature(const ElementType& ref_element, const T& alpha, const T& beta,
                              const std::size_t& max_newton_itersations = 10);

    private:
        const T alpha_m;
        const T beta_m;
    };

    template <typename T, unsigned NumNodes1D, typename ElementType>
    class GaussLegendreQuadrature : public GaussJacobiQuadrature<T, NumNodes1D, ElementType> {
    public:
        GaussLegendreQuadrature(const ElementType& ref_element,
                                const std::size_t& max_newton_itersations = 10)
            : GaussJacobiQuadrature<T, NumNodes1D, ElementType>(ref_element, 0.0, 0.0,
                                                                max_newton_itersations) {}
    };

    template <typename T, unsigned NumNodes1D, typename ElementType>
    class ChebyshevGaussQuadrature : public GaussJacobiQuadrature<T, NumNodes1D, ElementType> {
    public:
        ChebyshevGaussQuadrature(const ElementType& ref_element,
                                 const std::size_t& max_newton_itersations = 10)
            : GaussJacobiQuadrature<T, NumNodes1D, ElementType>(ref_element, -0.5, -0.5,
                                                                max_newton_itersations) {}
    };

}  // namespace ippl

#include "FEM/Quadrature/GaussJacobiQuadrature.hpp"

#endif