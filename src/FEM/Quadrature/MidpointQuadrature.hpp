

namespace ippl {
    template <typename T, unsigned NumNodes1D, typename ElementType>
    MidpointQuadrature<T, NumNodes1D, ElementType>::MidpointQuadrature(
        const ElementType& ref_element)
        : Quadrature<T, NumNodes1D, ElementType>(ref_element) {}

    template <typename T, unsigned NumNodes1D, typename ElementType>
    Vector<T, NumNodes1D> MidpointQuadrature<T, NumNodes1D, ElementType>::getIntegrationNodes()
        const {
        const unsigned number_of_segments = NumNodes1D;
        const T segment_length            = 1.0 / number_of_segments;

        // TODO use KOKKKOS
        Vector<T, NumNodes1D> nodes;
        T integration_point = 0.5 * segment_length;
        for (unsigned i = 0; i < number_of_segments; ++i) {
            nodes[i] = integration_point;
            integration_point += segment_length;
        }

        return nodes;
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    Vector<T, NumNodes1D> MidpointQuadrature<T, NumNodes1D, ElementType>::getWeights() const {
        return Vector<T, NumNodes1D>(1.0 / NumNodes1D);
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    std::size_t MidpointQuadrature<T, NumNodes1D, ElementType>::getDegree() const {
        return 1;
    }

}  // namespace ippl