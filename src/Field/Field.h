//
// Class Field
//   BareField with a mesh and configurable boundary conditions
//
#ifndef IPPL_FIELD_H
#define IPPL_FIELD_H

#include "Utility/TypeUtils.h"

#include "Field/BareField.h"
#include "Field/BConds.h"

#include "Meshes/UniformCartesian.h"

namespace ippl {

    template <typename T, unsigned Dim, class Mesh, class Centering, class... ViewArgs>
    class Field : public BareField<T, Dim, ViewArgs...> {
        template <typename... Props>
        using base_type = Field<T, Dim, Mesh, Centering, Props...>;

    public:
        using Mesh_t      = Mesh;
        using Centering_t = Cell;
        using Layout_t    = FieldLayout<Dim>;
        using BareField_t = BareField<T, Dim, ViewArgs...>;
        using view_type   = typename BareField_t::view_type;
        using BConds_t    = BConds<Field<T, Dim, Mesh, Centering, ViewArgs...>, Dim>;

        using uniform_type =
            typename detail::CreateUniformType<base_type, typename view_type::uniform_type>::type;

        // A default constructor, which should be used only if the user calls the
        // 'initialize' function before doing anything else.  There are no special
        // checks in the rest of the Field methods to check that the Field has
        // been properly initialized.
        Field();

        Field(const Field&) = default;

        /*!
         * Creates a new Field with the same properties and contents
         * @return A deep copy of the field
         */
        Field deepCopy() const;

        virtual ~Field() = default;

        // Constructors including a Mesh object as argument:
        Field(Mesh_t&, Layout_t&, int nghost = 1);

        // Initialize the Field, also specifying a mesh
        void initialize(Mesh_t&, Layout_t&, int nghost = 1);

        // ML
        void updateLayout(Layout_t&, int nghost = 1);

        void setFieldBC(BConds_t& bc) {
            bc_m = bc;
            bc_m.findBCNeighbors(*this);
        }

        // Access to the mesh
        KOKKOS_INLINE_FUNCTION Mesh_t& get_mesh() const { return *mesh_m; }

        /*!
         * Use the midpoint rule to calculate the field's volume integral
         * @return Integral of the field over its domain
         */
        T getVolumeIntegral() const;

        /*!
         * Use the midpoint rule to calculate the field's volume average
         * @return Integral of the field divided by the mesh volume
         */
        T getVolumeAverage() const;

        BConds_t& getFieldBC() { return bc_m; }
        // Assignment from constants and other arrays.
        using BareField<T, Dim, ViewArgs...>::operator=;

    private:
        // The Mesh object, and a flag indicating if we constructed it
        Mesh_t* mesh_m;

        // The boundary conditions.
        BConds_t bc_m;
    };
}  // namespace ippl

#include "Field/Field.hpp"
#include "Field/FieldOperations.hpp"

#endif

// vi: set et ts=4 sw=4 sts=4:
// Local Variables:
// mode:c
// c-basic-offset: 4
// indent-tabs-mode: nil
// require-final-newline: nil
// End:
