
namespace ippl {
    template <typename T, unsigned NumNodes1D, typename ElementType>
    Quadrature<T, NumNodes1D, ElementType>::Quadrature(const ElementType& ref_element)
        : ref_element_m(ref_element) {}

    template <typename T, unsigned NumNodes1D, typename ElementType>
    size_t Quadrature<T, NumNodes1D, ElementType>::getOrder() const {
        return this->degree_m + 1;
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    size_t Quadrature<T, NumNodes1D, ElementType>::getDegree() const {
        return this->degree_m;
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    Vector<T, Quadrature<T, NumNodes1D, ElementType>::numElementNodes>
    Quadrature<T, NumNodes1D, ElementType>::getWeightsForRefElement() const {
        Vector<T, NumNodes1D> w = this->getWeights1D(0.0, 1.0);

        Vector<T, std::remove_reference_t<decltype(*this)>::numElementNodes> tensor_prod_w;

        Vector<unsigned, ElementType::dim> nd_index(0);
        for (unsigned i = 0; i < std::remove_reference_t<decltype(*this)>::numElementNodes; ++i) {
            tensor_prod_w[i] = 1.0;
            for (unsigned d = 0; d < ElementType::dim; ++d) {
                tensor_prod_w[i] *= w[nd_index[d]];
            }

            // Update nd_index for next iteration
            // Increment the nd_index variable in the first dimension, or if it
            // is already at the maximum value reset it and, go to the higher dimension
            for (unsigned d = 0; d < ElementType::dim; ++d) {
                if (++nd_index[d] < NumNodes1D)
                    break;
                nd_index[d] = 0;
            }
        }

        return tensor_prod_w;
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    Vector<Vector<T, Quadrature<T, NumNodes1D, ElementType>::dim>,
           Quadrature<T, NumNodes1D, ElementType>::numElementNodes>
    Quadrature<T, NumNodes1D, ElementType>::getIntegrationNodesForRefElement() const {
        Vector<T, NumNodes1D> q = this->getIntegrationNodes1D(0.0, 1.0);

        Vector<Vector<T, ElementType::dim>, std::remove_reference_t<decltype(*this)>::numElementNodes> tensor_prod_q;

        Vector<unsigned, ElementType::dim> nd_index(0);
        for (unsigned i = 0; i < std::remove_reference_t<decltype(*this)>::numElementNodes; ++i) {
            for (unsigned d = 0; d < ElementType::dim; ++d) {
                tensor_prod_q[i][d] = q[nd_index[d]];
            }

            // Update nd_index for next iteration
            // Increment the nd_index variable in the first dimension, or if it
            // is already at the maximum value reset it and, go to the higher dimension
            for (unsigned d = 0; d < ElementType::dim; ++d) {
                if (++nd_index[d] < NumNodes1D)
                    break;
                nd_index[d] = 0;
            }
        }

        return tensor_prod_q;
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    Vector<T, NumNodes1D> Quadrature<T, NumNodes1D, ElementType>::getIntegrationNodes1D(
        const T& a, const T& b) const {
        assert(b > a);
        // scale the integration nodes from the local domain [a_m, b_m] to the given one [a, b]

        return (this->integration_nodes_m - this->a_m) / (this->b_m - this->a_m) * (b - a) + a;
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    Vector<T, NumNodes1D> Quadrature<T, NumNodes1D, ElementType>::getWeights1D(const T& a,
                                                                               const T& b) const {
        assert(b > a);
        return this->weights_m * (b - a) / (this->b_m - this->a_m);
    }

}  // namespace ippl
