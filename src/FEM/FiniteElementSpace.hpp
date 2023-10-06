
namespace ippl {
    template <typename T, unsigned Dim>
    FiniteElementSpace<T, Dim>::FiniteElementSpace(const Mesh<T, Dim>& mesh,
                                                   const Element<T, Dim>* ref_element,
                                                   const Quadrature<T>* quadrature,
                                                   const unsigned& degree)
        : mesh_m(mesh)
        , ref_element_m(ref_element)
        , quadrature_m(quadrature)
        , degree_m(degree) {
        assert(mesh.dim == Dim);
        assert(ref_element.dim == Dim);
    }

    template <typename T, unsigned Dim>
    void FiniteElementSpace<T, Dim>::setDegree(const unsigned& degree) {
        degree_m = degree;
    }

    template <typename T, unsigned Dim>
    unsigned FiniteElementSpace<T, Dim>::getDegree() const {
        return degree_m;
    }

    template <typename T, unsigned Dim>
    void FiniteElementSpace<T, Dim>::setOrder(const unsigned& order) {
        degree_m = order - 1;
    }

    template <typename T, unsigned Dim>
    unsigned FiniteElementSpace<T, Dim>::getOrder() const {
        return degree_m + 1;
    }

}  // namespace ippl