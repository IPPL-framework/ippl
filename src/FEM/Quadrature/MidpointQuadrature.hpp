

namespace ippl {
    template <typename T, unsigned NumNodes1D, typename ElementType>
    MidpointQuadrature<T, NumNodes1D, ElementType>::MidpointQuadrature(
        const ElementType& ref_element)
        : Quadrature<T, NumNodes1D, ElementType>(ref_element) {
        this->degree_m = 1;

        this->a_m = 0.0;
        this->b_m = 1.0;

        computeNodesAndWeights();
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    void MidpointQuadrature<T, NumNodes1D, ElementType>::computeNodesAndWeights() {
        const T segment_length = (this->b_m - this->a_m) / NumNodes1D;

        this->weights_m = Vector<T, NumNodes1D>(segment_length);

        this->integration_nodes_m = Vector<T, NumNodes1D>();
        for (unsigned i = 0; i < NumNodes1D; ++i) {
            this->integration_nodes_m[i] = 0.5 * segment_length + i * segment_length + this->a_m;
        }
    }

}  // namespace ippl