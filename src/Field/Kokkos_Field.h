#ifndef Kokkos_Field_H
#define Kokkos_Field_H

#include "Field/BareField.h"
#include "Meshes/Kokkos_UniformCartesian.h"

namespace ippl {

    template<typename T, unsigned Dim,
            class M=UniformCartesian<Dim,double>,
            class C=typename M::DefaultCentering >
    class Kokkos_Field : public BareField<T, Dim> {

    public:
        typedef M Mesh_t;
        typedef C Centering_t;
          typedef FieldLayout<Dim>                   Layout_t;

        // A default constructor, which should be used only if the user calls the
        // 'initialize' function before doing anything else.  There are no special
        // checks in the rest of the Kokkos_Field methods to check that the Kokkos_Field has
        // been properly initialized.
        Kokkos_Field();

        // Destroy the Kokkos_Field.
        virtual ~Kokkos_Field();

        // Create a new Kokkos_Field with a given layout and optional guard cells.
        // The default type of BCond lets you add new ones dynamically.
        // The makeMesh() global function is a way to allow for different types of
        // constructor arguments for different mesh types.
        Kokkos_Field(Layout_t &);

        // Constructors including a Mesh object as argument:
        Kokkos_Field(Mesh_t&, Layout_t &);

        // Initialize the Kokkos_Field, if it was constructed from the default constructor.
        // This should NOT be called if the Kokkos_Field was constructed by providing
        // a Kokkos_FieldLayout or FieldSpec
        void initialize(Layout_t &);

        // Initialize the Kokkos_Field, also specifying a mesh
        void initialize(Mesh_t&, Layout_t &);

        // Access to the mesh
        Mesh_t& get_mesh() const { return *mesh; }

        // Assignment from constants and other arrays.
        const Kokkos_Field<T, Dim, M, C>& operator=(T x) {
            assign(*this,x);
            return *this;
        }

        const Kokkos_Field<T, Dim, M, C>& operator=(const Kokkos_Field<T, Dim, M, C>& x) {
            assign(*this,x);
            return *this;
        }

    private:
        //   // The boundary conditions.
        //   bcond_container Bc;

        // The Mesh object, and a flag indicating if we constructed it
        Mesh_t* mesh;
        bool WeOwnMesh;

        // store the given mesh object pointer, and the flag whether we own it or not.
        // if we own it, we must make sure to delete it when this Kokkos_Field is deleted.
        void store_mesh(Mesh_t*, bool);

        // delete the mesh object, if necessary; otherwise, just zero the pointer
        void delete_mesh();
        };
}

#include "Field/Kokkos_Field.hpp"

#endif

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
