

namespace ippl {
    template <typename T, unsigned NumNodes1D, typename ElementType>
    MidpointQuadrature<T, NumNodes1D, ElementType>::MidpointQuadrature(
        const ElementType& ref_element)
        : Quadrature<T, NumNodes1D, ElementType>(ref_element) {
        this->degree_m = 1;

        this->weights_m = Vector<T, NumNodes1D>(1.0 / NumNodes1D);

        const unsigned number_of_segments = NumNodes1D;
        const T segment_length            = 1.0 / number_of_segments;

        // TODO use KOKKKOS
        this->integration_nodes_m = Vector<T, NumNodes1D>();
        T integration_point       = 0.5 * segment_length;
        for (unsigned i = 0; i < number_of_segments; ++i) {
            this->integration_nodes_m[i] = integration_point;
            this->integration_nodes_m[i] += segment_length;
        }
    }

}  // namespace ippl