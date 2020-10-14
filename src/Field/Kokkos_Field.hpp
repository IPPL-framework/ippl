namespace ippl {

    // Generic makeMesh function
    template<typename T, unsigned Dim, class M, class C>
    inline std::shared_ptr<M> makeMesh(Field<T, Dim, M, C>& f)
    {
        NDIndex<Dim> ndi;
        ndi = f.getLayout().getDomain();
        return std::make_shared<M>(ndi);
    }

    // Specialization for UniformCartesian
    template<class T, unsigned Dim, class C>
    std::shared_ptr<UniformCartesian<T, Dim>> makeMesh(Field<T,Dim,UniformCartesian<T, Dim>,C>& f)
    {
        NDIndex<Dim> ndi;
        ndi = f.getLayout().getDomain();
        return std::make_shared<UniformCartesian<T, Dim>>(ndi);
    }

    /*
    // Specialization for Cartesian
    template<class T, unsigned Dim, class MFLOAT, class C>
    Cartesian<Dim,MFLOAT>*
    makeMesh(Field<T,Dim,Cartesian<Dim,MFLOAT>,C>& f)
    {



    NDIndex<Dim> ndi;
    ndi = f.getLayout().getDomain();
    return (new Cartesian<Dim,MFLOAT>(ndi));
    }
    */

    //=============================================================================
    // Field member functions
    //=============================================================================


    //////////////////////////////////////////////////////////////////////////
    // A default constructor, which should be used only if the user calls the
    // 'initialize' function before doing anything else.  There are no special
    // checks in the rest of the Field methods to check that the Field has
    // been properly initialized
    template<class T, unsigned Dim, class M, class C>
    Field<T,Dim,M,C>::Field() : BareField<T, Dim>() {
        storeMesh_m(nullptr, true);
    }


    //////////////////////////////////////////////////////////////////////////
    // Create a new Field with a given layout and optional guard cells.
    // The default type of BCond lets you add new ones dynamically.
    // The makeMesh() global function is a way to allow for different types of
    // constructor arguments for different mesh types.
    template<class T, unsigned Dim, class M, class C>
    Field<T,Dim,M,C>::Field(Layout_t& l) : BareField<T,Dim>(l) {
        storeMesh_m(makeMesh(*this), true);
    }

    //////////////////////////////////////////////////////////////////////////
    // Constructors which include a Mesh object as argument
    template<class T, unsigned Dim, class M, class C>
    Field<T,Dim,M,C>::Field(Mesh_t& m, Layout_t & l)
        : BareField<T,Dim>(l)
    {
        storeMesh_m(std::make_shared<Mesh_t>(m), false);
    }

    //////////////////////////////////////////////////////////////////////////
    // Initialize the Field, if it was constructed from the default constructor.
    // This should NOT be called if the Field was constructed by providing
    // a FieldLayout or FieldSpec
    template<class T, unsigned Dim, class M, class C>
    void Field<T,Dim,M,C>::initialize(Layout_t & l) {
        BareField<T,Dim>::initialize(l);
        storeMesh_m(makeMesh(*this), true);
    }




    //////////////////////////////////////////////////////////////////////////
    // Initialize the Field, also specifying a mesh
    template<class T, unsigned Dim, class M, class C>
    void Field<T,Dim,M,C>::initialize(Mesh_t& m, Layout_t & l) {
        BareField<T,Dim>::initialize(l);
        storeMesh_m(std::make_shared<Mesh_t>(m), false);
    }


    //////////////////////////////////////////////////////////////////////////
    // store the given mesh object pointer, and the flag whether we use it or not
    template<class T, unsigned Dim, class M, class C>
    void Field<T,Dim,M,C>::storeMesh_m(const std::shared_ptr<Mesh_t>& m, bool WeOwn) {
        mesh = m;
        WeOwnMesh_m = WeOwn;
    }
}
