
namespace ippl {

    template <typename T, unsigned NumNodes, unsigned Dim, unsigned NumElementVertices>
    std::size_t Quadrature<T, NumNodes, NumElementVertices>::num1DIntegrationPoints() const {
        return NumNodes;
    }

    template <typename T, unsigned NumNodes, unsigned Dim, unsigned NumElementVertices>
    std::size_t Quadrature<T, NumNodes, NumElementVertices>::numElementIntegrationPoints() const {
        return numElementNodes(NumNodes, Dim);
    }

    template <typename T, unsigned NumNodes, unsigned Dim, unsigned NumElementVertices>
    std::size_t Quadrature<T, NumNodes, NumElementVertices>::getOrder() const {
        return this->getDegree() + 1;
    }

    template <typename T, unsigned NumNodes, unsigned Dim, unsigned NumElementVertices>
    Vector<T, numElementNodes(NumNodes, Dim)>
    Quadrature<T, NumNodes, Dim, NumElementVertices>::getWeightsForRefElement() const {
        const std::size_t NumElementNodes = numElementNodes(NumNodes, Dim);

        Vector<T, NumNodes> w = this->getWeights();

        Vector<T, NumElementNodes> tensor_prod_w;

        Vector<unsigned, Dim> nd_index(0);
        for (unsigned i = 0; i < NumElementNodes; ++i) {
            tensor_prod_w[i] = 1.0;
            for (unsigned d = 0; d < Dim; ++d) {
                tensor_prod_w[i] *= w[nd_index[d]];
            }

            // Update nd_index for next iteration
            // Increment the nd_index variable in the first dimension, or if it
            // is already at the maximum value reset it and, go to the higher dimension
            for (int d = 0; d < Dim; ++d) {
                if (++nd_index[d] < NumNodes)
                    break;
                nd_index[d] = 0;
            }
        }

        return tensor_prod_w;
    }

    template <typename T, unsigned NumNodes, unsigned Dim, unsigned NumElementVertices>
    Vector<Vector<T, Dim>, numElementNodes(NumNodes, Dim)>
    Quadrature<T, NumNodes, NumElementVertices>::getIntegrationNodesForRefElement() const {
        const std::size_t NumElementNodes = numElementNodes(NumNodes, Dim);

        Vector<T, NumNodes> q = this->getIntegrationNodes();

        Vector<Vector<T, Dim>, NumElementNodes> tensor_prod_q;

        Vector<unsigned, Dim> nd_index(0);
        for (unsigned i = 0; i < NumElementNodes; ++i) {
            for (unsigned d = 0; d < Dim; ++d) {
                tensor_prod_q[i][d] = q[nd_index[d]];
            }

            // Update nd_index for next iteration
            // Increment the nd_index variable in the first dimension, or if it
            // is already at the maximum value reset it and, go to the higher dimension
            for (int d = 0; d < Dim; ++d) {
                if (++nd_index[d] < NumNodes)
                    break;
                nd_index[d] = 0;
            }
        }

        return tensor_prod_q;
    }

}  // namespace ippl
