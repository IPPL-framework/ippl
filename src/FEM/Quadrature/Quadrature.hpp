

namespace ippl {
    template <typename T>
    Quadrature<T>::Quadrature(const unsigned& degree)
        : degree_m(degree) {}

    template <typename T>
    unsigned Quadrature<T>::getOrder() const {
        return getDegree() + 1;
    }

    template <typename T>
    unsigned Quadrature<T>::getDegree() const {
        return degree_m;
    }

    template <typename T>
    void Quadrature<T>::setOrder(const unsigned& order) {
        assert(order > 1 && "Order must be greater than 1.");
        degree_m = order - 1;
    }

    template <typename T>
    void Quadrature<T>::setDegree(const unsigned& degree) {
        assert(degree > 0 && "Degree must be greater than 0.");
        degree_m = degree;
    }

}  // namespace ippl
