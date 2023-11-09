
namespace ippl {
    template <typename T, unsigned NumNodes1D, typename ElementType>
    std::size_t Quadrature<T, NumNodes1D, ElementType>::getOrder() const {
        return this->getDegree() + 1;
    }

    template <typename T, unsigned NumNodes1D, typename ElementType>
    Vector<T, Quadrature<T, NumNodes1D, ElementType>::numElementNodes>
    Quadrature<T, NumNodes1D, ElementType>::getWeightsForRefElement() const {
        Vector<T, NumNodes1D> w = this->getWeights();

        Vector<T, this->numElementNodes> tensor_prod_w;

        Vector<unsigned, ElementType::dim> nd_index(0);
        for (unsigned i = 0; i < this->numElementNodes; ++i) {
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
        Vector<T, NumNodes1D> q = this->getIntegrationNodes();

        Vector<Vector<T, ElementType::dim>, this->numElementNodes> tensor_prod_q;

        Vector<unsigned, ElementType::dim> nd_index(0);
        for (unsigned i = 0; i < this->numElementNodes; ++i) {
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

}  // namespace ippl
