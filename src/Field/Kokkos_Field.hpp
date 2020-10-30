namespace ippl {

    template<class T, unsigned Dim, class Mesh, class Cell>
    Field<T, Dim, Mesh, Cell>::Field()
    : BareField<T, Dim>()
    , mesh_m(nullptr)
    { }


    template<class T, unsigned Dim, class Mesh, class Cell>
    Field<T, Dim, Mesh, Cell>::Field(Mesh_t& m, Layout_t& l)
    : BareField<T,Dim>(l)
    , mesh_m(&m)
    { }


    template<class T, unsigned Dim, class Mesh, class Cell>
    Field<T, Dim, Mesh, Cell>::Field(Mesh_t& m, Layout_t& l,
                                          const bc_container& bc)
    : Field<T,Dim>(m, l)
    {
        for (unsigned int i = 0; i < 2 * Dim; ++i) {
            bc_m[i] = bc[i];
        }
    }


    template<class T, unsigned Dim, class Mesh, class Cell>
    void Field<T, Dim, Mesh, Cell>::initialize(Mesh_t& m, Layout_t& l)
    {
        BareField<T,Dim>::initialize(l);
        mesh_m = &m;
    }


    template<class T, unsigned Dim, class Mesh, class Cell>
    void Field<T, Dim, Mesh, Cell>::initialize(Mesh_t& m, Layout_t& l,
                                                    const bc_container& bc)
    {
        initialize(m, l);
        bc_m = bc;
    }
}
