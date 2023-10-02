

namespace ippl {
    template <typename T, unsigned Dim, unsigned NumNodes>
    GaussJacobiQuadrature<T, Dim, NumNodes>::GaussJacobiQuadrature(const T& alpha = 0.0,
                                                                   const T& beta  = 0.0)
        : Quadrature<T, Dim>()
        , alpha_m(alpha)
        , beta_m(beta) {}

    template <typename T, unsigned Dim, unsigned NumNodes>
    Vector<Vector<T, Dim>, NumNodes> GaussJacobiQuadrature<T, Dim, NumNodes>::getNodes(
        const T& a = -1.0, const T& b = 1.0) const {
        return {}  // TODO
    }

    template <typename T, unsigned Dim, unsigned NumNodes>
    Vector<T, NumNodes> GaussJacobiQuadrature<T, Dim, NumNodes>::getWeights() const {
        return {}  // TODO
    }

    template <typename T, unsigned Dim, unsigned NumNodes>
    unsigned GaussJacobiQuadrature<T, Dim, NumNodes>::getNumberOfPoints() const {
        return number_of_points_m;
    }
}  // namespace ippl