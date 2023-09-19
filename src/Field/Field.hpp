//
// Class Field
//   BareField with a mesh and configurable boundary conditions
//
//

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim, class Mesh, class Centering, class... ViewArgs>
        struct isExpression<Field<T, Dim, Mesh, Centering, ViewArgs...>> : std::true_type {};
    }  // namespace detail

    //////////////////////////////////////////////////////////////////////////
    // A default constructor, which should be used only if the user calls the
    // 'initialize' function before doing anything else.  There are no special
    // checks in the rest of the Field methods to check that the Field has
    // been properly initialized
    template <class T, unsigned Dim, class Mesh, class Centering, class... ViewArgs>
    Field<T, Dim, Mesh, Centering, ViewArgs...>::Field()
        : BareField_t()
        , mesh_m(nullptr)
        , bc_m() {}

    template <class T, unsigned Dim, class Mesh, class Centering, class... ViewArgs>
    Field<T, Dim, Mesh, Centering, ViewArgs...>
    Field<T, Dim, Mesh, Centering, ViewArgs...>::deepCopy() const {
        Field<T, Dim, Mesh, Centering, ViewArgs...> copy(*mesh_m, this->getLayout(),
                                                         this->getNghost());
        Kokkos::deep_copy(copy.getView(), this->getView());

        return copy;
    }

    //////////////////////////////////////////////////////////////////////////
    // Constructors which include a Mesh object as argument
    template <class T, unsigned Dim, class Mesh, class Centering, class... ViewArgs>
    Field<T, Dim, Mesh, Centering, ViewArgs...>::Field(Mesh_t& m, Layout_t& l, int nghost)
        : BareField_t(l, nghost)
        , mesh_m(&m) {
        for (unsigned int face = 0; face < 2 * Dim; ++face) {
            bc_m[face] =
                std::make_shared<NoBcFace<Field<T, Dim, Mesh, Centering, ViewArgs...>>>(face);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // Initialize the Field, also specifying a mesh
    template <class T, unsigned Dim, class Mesh, class Centering, class... ViewArgs>
    void Field<T, Dim, Mesh, Centering, ViewArgs...>::initialize(Mesh_t& m, Layout_t& l,
                                                                 int nghost) {
        BareField_t::initialize(l, nghost);
        mesh_m = &m;
        for (unsigned int face = 0; face < 2 * Dim; ++face) {
            bc_m[face] =
                std::make_shared<NoBcFace<Field<T, Dim, Mesh, Centering, ViewArgs...>>>(face);
        }
    }

    template <class T, unsigned Dim, class Mesh, class Centering, class... ViewArgs>
    T Field<T, Dim, Mesh, Centering, ViewArgs...>::getVolumeIntegral() const {
        typename Mesh::value_type dV = mesh_m->getCellVolume();
        return this->sum() * dV;
    }

    template <class T, unsigned Dim, class Mesh, class Centering, class... ViewArgs>
    T Field<T, Dim, Mesh, Centering, ViewArgs...>::getVolumeAverage() const {
        return getVolumeIntegral() / mesh_m->getMeshVolume();
    }

    template <class T, unsigned Dim, class Mesh, class Centering, class... ViewArgs>
    void Field<T, Dim, Mesh, Centering, ViewArgs...>::updateLayout(Layout_t& l, int nghost) {
        BareField_t::updateLayout(l, nghost);
    }

}  // namespace ippl
