

namespace ippl {

    template <typename T, unsigned NumNodes>
    unsigned Quadrature<T, NumNodes>::getNumberOfIntegrationPoints() const {
        return NumNodes;
    }

}  // namespace ippl
