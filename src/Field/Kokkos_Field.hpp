namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim, class M, class C>
        struct isFieldExpression<Field<T, Dim, M, C>> : std::true_type {};
    }

    //////////////////////////////////////////////////////////////////////////
    // A default constructor, which should be used only if the user calls the
    // 'initialize' function before doing anything else.  There are no special
    // checks in the rest of the Field methods to check that the Field has
    // been properly initialized
    template<class T, unsigned Dim, class M, class C>
    Field<T,Dim,M,C>::Field()
    : BareField<T, Dim>()
    , mesh_m(nullptr)
    { }

    //////////////////////////////////////////////////////////////////////////
    // Constructors which include a Mesh object as argument
    template<class T, unsigned Dim, class M, class C>
    Field<T,Dim,M,C>::Field(Mesh_t& m, Layout_t& l)
    : BareField<T,Dim>(l)
    , mesh_m(&m)
    { }


    //////////////////////////////////////////////////////////////////////////
    // Initialize the Field, also specifying a mesh
    template<class T, unsigned Dim, class M, class C>
    void Field<T,Dim,M,C>::initialize(Mesh_t& m, Layout_t& l) {
        BareField<T,Dim>::initialize(l);
        mesh_m = &m;
    }


//     template<class T, unsigned Dim, class M, class C>
//     void Field<T,Dim, M, C>::write(std::ostream& out) const {
//         barefield_m.write(out);
//     }
}
