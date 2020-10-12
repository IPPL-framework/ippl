#ifndef IPPL_FIELD_H
#define IPPL_FIELD_H

#include "Field/BareField.h"
#include "Meshes/Kokkos_UniformCartesian.h"

namespace ippl {

    template<typename T, unsigned Dim,
            class M=UniformCartesian<T, Dim>,
            class C=typename M::DefaultCentering >
    class Field : public BareField<T, Dim> {

    public:
        typedef M Mesh_t;
        typedef C Centering_t;
        typedef FieldLayout<Dim> Layout_t;

        // A default constructor, which should be used only if the user calls the
        // 'initialize' function before doing anything else.  There are no special
        // checks in the rest of the Field methods to check that the Field has
        // been properly initialized.
        Field();

        virtual ~Field() = default;

        // Create a new Field with a given layout and optional guard cells.
        // The default type of BCond lets you add new ones dynamically.
        // The makeMesh() global function is a way to allow for different types of
        // constructor arguments for different mesh types.
        Field(Layout_t&);

        // Constructors including a Mesh object as argument:
        Field(Mesh_t&, Layout_t&);

        // Initialize the Field, if it was constructed from the default constructor.
        // This should NOT be called if the Field was constructed by providing
        // a FieldLayout or FieldSpec
        void initialize(Layout_t&);

        // Initialize the Field, also specifying a mesh
        void initialize(Mesh_t&, Layout_t &);

        // Access to the mesh
        Mesh_t& get_mesh() const { return *mesh; }

        // Assignment from constants and other arrays.
        using BareField<T, Dim>::operator=;

    private:
        //   // The boundary conditions.
        //   bcond_container Bc;

        // The Mesh object, and a flag indicating if we constructed it
        std::shared_ptr<Mesh_t> mesh;
        bool WeOwnMesh_m;

        // store the given mesh object pointer, and the flag whether we own it or not.
        // if we own it, we must make sure to delete it when this Field is deleted.
        void storeMesh_m(const std::shared_ptr<Mesh_t>&, bool);
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
