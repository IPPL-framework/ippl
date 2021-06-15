namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim, class M, class C>
        struct isExpression<Field<T, Dim, M, C>> : std::true_type {};
    }

template<class T, unsigned Dim, class M, class C>
    Field<T,Dim,M,C>::Field()
    : BareField<T, Dim>()
    , mesh_m(nullptr)
    , bc_m() 
    { }

    template<class T, unsigned Dim, class M, class C>
    Field<T,Dim,M,C>::Field(Mesh_t& m, Layout_t& l, int nghost)
    : BareField<T,Dim>(l, nghost)
    , mesh_m(&m)
    { 
        for (unsigned int face=0; face < 2 * Dim; ++face) {
            bc_m[face] = std::make_shared<NoBcFace<T, Dim>>(face);
        }
    }

    template<class T, unsigned Dim, class M, class C>
    void Field<T,Dim,M,C>::initialize(Mesh_t& m, Layout_t& l, int nghost) {
        BareField<T,Dim>::initialize(l, nghost);
        mesh_m = &m;
        for (unsigned int face=0; face < 2 * Dim; ++face) {
            bc_m[face] = std::make_shared<NoBcFace<T, Dim>>(face);
        }
    }

    template<class T, unsigned Dim, class M, class C>
    T Field<T,Dim,M,C>::getVolumeIntegral() const {
        typename M::value_type dV = mesh_m->getCellVolume();
        return this->sum() * dV;
    }

    template<class T, unsigned Dim, class M, class C>
    T Field<T,Dim,M,C>::getVolumeAverage() const {
        return getVolumeIntegral() / mesh_m->getMeshVolume();
    }

    template<class T, unsigned Dim, class M, class C>
    void Field<T,Dim,M,C>::updateLayout(Layout_t& l, int nghost) {
        BareField<T,Dim>::updateLayout(l, nghost);
    }

}

