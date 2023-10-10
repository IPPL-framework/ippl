

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
        const number_of_segments = num_integration_points_m;
        const segment_length     = (b - a) / number_of_segments;

        // TODO use KOKKKOS
        Vector<T, NumNodes> nodes;
        T integration_point = a + 0.5 * segment_length;
        for (unsigned i = 0; i < number_of_segments; ++i) {
            nodes[i] = integration_point;
            integration_point += segment_length;
        }

        return nodes;
    }

    template <typename T>
    template <unsigned NumNodes>
    Vector<T, NumNodes> MidpointQuadrature<T>::getWeights() const {
        const T interval_length = b - a;
        return Vector<T, NumNodes>(interval_length / num_integration_points_m);
    }

    template <typename T>
    unsigned MidpointQuadrature<T>::getDegree() const {
        return 1;
    }

}  // namespace ippl