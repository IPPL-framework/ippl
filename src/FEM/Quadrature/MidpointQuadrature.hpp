

namespace ippl {
    template <typename T>
    MidpointQuadrature<T>::MidpointQuadrature(const unsigned& num_integration_points = 1)
        : num_integration_points_m(num_integration_points) {
        assert(num_integration_points >= 1 && "Number of integration points must be at least 1.");
    }

    template <typename T>
    unsigned MidpointQuadrature<T>::getNumberOfIntegrationPoints() const {
        return num_integration_points_m;
    }

    template <typename T>
    template <unsigned NumNodes>
    Vector<T, NumNodes> MidpointQuadrature<T>::getIntegrationNodes(const T& a, const T& b) const {
        // return {a + ((b - a) / 2.0)};
    }

    template <typename T>
    template <unsigned NumNodes>
    Vector<T, NumNodes> MidpointQuadrature<T>::getWeights() const {
                return Vector<T, NumNodes>(1.0 / num_integration_points_m);  // TODO
    }

    template <typename T>
    unsigned MidpointQuadrature<T>::getDegree() const {
        return 1;
    }

}  // namespace ippl