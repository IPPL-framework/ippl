

namespace ippl {
    template <typename T>
    MidpointQuadrature<T>::MidpointQuadrature() {}

    template <typename T>
    Vector<T, 1> MidpointQuadrature<T>::getNodes(const T& a, const T& b) const {
        return {a + ((b - a) / 2.0)};
    }

    template <typename T>
    Vector<T, 1> MidpointQuadrature<T>::getWeights() const {
        return {1.0};
    }

}  // namespace ippl