
#include <cmath>

namespace ippl {
    template <typename T>
    GaussJacobiQuadrature<T>::GaussJacobiQuadrature(const unsigned& degree, const T& alpha,
                                                    const T& beta)
        : Quadrature<T>(degree, a, b)
        , alpha_m(alpha)
        , beta_m(beta) {}

    template <typename T>
    unsigned GaussJacobiQuadrature<T>::getNumberOfIntegrationPoints() const {
        return std::ceil((getDegree() + 1) / 2);  // TODO
    }
    template <typename T>
    template <unsigned NumNodes>
    Vector<T, NumNodes> GaussJacobiQuadrature<T>::getIntegrationNodes(const T& a = -1.0,
                                                                      const T& b = 1.0) const {
        return {}  // TODO
    }

    template <typename T>
    template <unsigned NumNodes>
    Vector<T, NumNodes> GaussJacobiQuadrature<T>::getWeights() const {
        return {}  // TODO
    }
}  // namespace ippl