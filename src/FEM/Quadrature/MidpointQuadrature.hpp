

namespace ippl {
    template <typename T, unsigned NumIntegrationPoints>
    Vector<T, NumIntegrationPoints>
    MidpointQuadrature<T, NumIntegrationPoints>::getIntegrationNodes(const T& a, const T& b) const {
        const unsigned number_of_segments = this->getNumberOfIntegrationPoints();
        const T segment_length            = (b - a) / number_of_segments;

        // TODO use KOKKKOS
        Vector<T, number_of_segments> nodes;
        T integration_point = a + 0.5 * segment_length;
        for (unsigned i = 0; i < number_of_segments; ++i) {
            nodes[i] = integration_point;
            integration_point += segment_length;
        }

        return nodes;
    }

    template <typename T, unsigned NumIntegrationPoints>
    Vector<T, NumIntegrationPoints> MidpointQuadrature<T, NumIntegrationPoints>::getWeights(
        const T& a, const T& b) const {
        const T interval_length = b - a;
        return Vector<T, NumIntegrationPoints>(interval_length / NumIntegrationPoints);
    }

    template <typename T, unsigned NumIntegrationPoints>
    unsigned MidpointQuadrature<T, NumIntegrationPoints>::getDegree() const {
        return 1;
    }

}  // namespace ippl