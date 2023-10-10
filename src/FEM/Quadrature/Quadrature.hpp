

namespace ippl {

    template <typename T, unsigned NumNodes>
    unsigned Quadrature<T, NumNodes>::getNumberOfIntegrationPoints() const {
        return NumNodes;
    }

    template <typename T, unsigned NumNodes>
    unsigned Quadrature<T, NumNodes>::getOrder() const {
        return this->getDegree() + 1;
    }

}  // namespace ippl
