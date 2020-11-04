#ifndef IPPL_FIELD_H
#define IPPL_FIELD_H

#include "Field/BareField.h"
#include "Meshes/Kokkos_UniformCartesian.h"

namespace ippl {

    template <typename T, unsigned Dim,
              class M=UniformCartesian<double, Dim>,
              class C=typename M::DefaultCentering >
    class Field : public detail::FieldExpression<
                            Field<T, Dim, M, C>,
                            sizeof(typename BareField<T, Dim>::view_type)>
                , public BareField<T, Dim>
    {
    public:
        using Mesh_t      = M;
        using Centering_t = C;
        using Layout_t    = FieldLayout<Dim>;
        using BareField_t = BareField<T, Dim>;
        using view_type   = typename BareField_t::view_type;

        // A default constructor, which should be used only if the user calls the
        // 'initialize' function before doing anything else.  There are no special
        // checks in the rest of the Field methods to check that the Field has
        // been properly initialized.
        Field();

        virtual ~Field() = default;

        // Constructors including a Mesh object as argument:
        Field(Mesh_t&, Layout_t&);

        // Initialize the Field, also specifying a mesh
        void initialize(Mesh_t&, Layout_t&);

        // Access to the mesh
        Mesh_t& get_mesh() const { return *mesh_m; }

        // Assignment from constants and other arrays.
        using BareField<T, Dim>::operator=;

        Field(const Field&) = default;

    private:
        //   // The boundary conditions.
        //   bcond_container Bc;

        // The Mesh object, and a flag indicating if we constructed it
        Mesh_t* mesh_m;
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
