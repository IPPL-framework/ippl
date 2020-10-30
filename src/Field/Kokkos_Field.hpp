namespace ippl {

    //////////////////////////////////////////////////////////////////////////
    // A default constructor, which should be used only if the user calls the
    // 'initialize' function before doing anything else.  There are no special
    // checks in the rest of the Field methods to check that the Field has
    // been properly initialized
    template<class T, unsigned Dim, class Mesh, class Centering>
    Field<T, Dim, Mesh, Centering>::Field()
    : BareField<T, Dim>()
    , mesh_m(nullptr)
    { }

    //////////////////////////////////////////////////////////////////////////
    // Constructors which include a Mesh object as argument
    template<class T, unsigned Dim, class Mesh, class Centering>
    Field<T, Dim, Mesh, Centering>::Field(Mesh_t& m, Layout_t& l)
        : BareField<T,Dim>(l)
        , mesh_m(&m)
    { }


    //////////////////////////////////////////////////////////////////////////
    // Initialize the Field, also specifying a mesh
    template<class T, unsigned Dim, class Mesh, class Centering>
    void Field<T, Dim, Mesh, Centering>::initialize(Mesh_t& m, Layout_t& l) {
        BareField<T,Dim>::initialize(l);
        mesh_m = &m;
    }
}
