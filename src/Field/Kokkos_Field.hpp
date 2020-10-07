#include "Field/Kokkos_Field.h"
// #include "Index/SIndex.h"


//=============================================================================
// Global functions
//=============================================================================

//-----------------------------------------------------------------------------
// Wrappers for internal mesh object constructor calls:
//-----------------------------------------------------------------------------
// Generic case, report unimplemented mesh type:
// (Compiler limitations doesn't allow this yet; user will get an obscure
// compile error instead of this runtime error if he specifies an unsupported
// mesh type for the Field Mesh parameter --tjw):
// template<class T, unsigned Dim, class M, class C>
// M* makeMesh(Kokkos_Field<T,Dim,M,C>& f)
// {
//   ERRORMSG("makeMesh() invoked from Field(): unimplemented mesh type" << endl);
// }


namespace ippl {

    // Generic makeMesh function
    template<class T, unsigned Dim, class M, class C>
    inline M* makeMesh(Kokkos_Field<T,Dim,M,C>& f)
    {
        NDIndex<Dim> ndi;
        ndi = f.getLayout().getDomain();
        return (new M(ndi));
    }

    /*
    // Specialization for UniformCartesian
    template<class T, unsigned Dim, class MFLOAT, class C>
    UniformCartesian<Dim,MFLOAT>*
    makeMesh(Kokkos_Field<T,Dim,UniformCartesian<Dim,MFLOAT>,C>& f)
    {



    NDIndex<Dim> ndi;
    ndi = f.getLayout().getDomain();
    return (new UniformCartesian<Dim,MFLOAT>(ndi));
    }

    // Specialization for Cartesian
    template<class T, unsigned Dim, class MFLOAT, class C>
    Cartesian<Dim,MFLOAT>*
    makeMesh(Kokkos_Field<T,Dim,Cartesian<Dim,MFLOAT>,C>& f)
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
    Kokkos_Field<T,Dim,M,C>::Kokkos_Field() {
        store_mesh(0, true);
    }


    //////////////////////////////////////////////////////////////////////////
    // Field destructor
    template<class T, unsigned Dim, class M, class C>
    Kokkos_Field<T,Dim,M,C>::~Kokkos_Field() {
        delete_mesh();
    }


    //////////////////////////////////////////////////////////////////////////
    // Create a new Field with a given layout and optional guard cells.
    // The default type of BCond lets you add new ones dynamically.
    // The makeMesh() global function is a way to allow for different types of
    // constructor arguments for different mesh types.
    template<class T, unsigned Dim, class M, class C>
    Kokkos_Field<T,Dim,M,C>::Kokkos_Field(Layout_t & l) : BareField<T,Dim>(l) {
        store_mesh(makeMesh(*this), true);
    }

    //////////////////////////////////////////////////////////////////////////
    // Constructors which include a Mesh object as argument
    template<class T, unsigned Dim, class M, class C>
    Kokkos_Field<T,Dim,M,C>::Kokkos_Field(Mesh_t& m, Layout_t & l)
        : BareField<T,Dim>(l)
    {
        store_mesh(&m, false);
    }

    //////////////////////////////////////////////////////////////////////////
    // Initialize the Field, if it was constructed from the default constructor.
    // This should NOT be called if the Field was constructed by providing
    // a FieldLayout or FieldSpec
    template<class T, unsigned Dim, class M, class C>
    void Kokkos_Field<T,Dim,M,C>::initialize(Layout_t & l) {
        BareField<T,Dim>::initialize(l);
        store_mesh(makeMesh(*this), true);
    }




    //////////////////////////////////////////////////////////////////////////
    // Initialize the Field, also specifying a mesh
    template<class T, unsigned Dim, class M, class C>
    void Kokkos_Field<T,Dim,M,C>::initialize(Mesh_t& m, Layout_t & l) {
        BareField<T,Dim>::initialize(l);
        store_mesh(&m, false);
    }


    //////////////////////////////////////////////////////////////////////////
    // store the given mesh object pointer, and the flag whether we use it or not
    template<class T, unsigned Dim, class M, class C>
    void Kokkos_Field<T,Dim,M,C>::store_mesh(Mesh_t* m, bool WeOwn) {
        mesh = m;
        WeOwnMesh = WeOwn;
        //FIXME
//         if (mesh != 0)
//             mesh->checkin(*this);
    }


    //////////////////////////////////////////////////////////////////////////
    // delete the mesh object, if necessary; otherwise, just zero the pointer
    template<class T, unsigned Dim, class M, class C>
    void Kokkos_Field<T,Dim,M,C>::delete_mesh() {
        if (mesh != 0) {
            //FIXME
//             mesh->checkout(*this);
            if (WeOwnMesh)
            delete mesh;
            mesh = 0;
        }
    }
}
