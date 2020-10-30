#ifndef IPPL_FIELD_BC_TYPES_H
#define IPPL_FIELD_BC_TYPES_H

namespace ippl {
    namespace detail {
        template<typename T, unsigned Dim, class Mesh, class Cell> class BCondBase;

        template<typename T, unsigned Dim, class Mesh, class Cell>
        std::ostream& operator<<(std::ostream&, const BCondBase<T, Dim, Mesh, Cell>&);


        template<typename T, unsigned Dim, class Mesh, class Cell>
        class BCondBase
        {
        public:
            // Special value designating application to all components of elements:
            static int allComponents;

            // Constructor takes:
            // face: the face to apply the boundary condition on.
            // i : what component of T to apply the boundary condition to.
            // The components default to setting all components.
            BCondBase(unsigned int face, int i = allComponents);

            virtual ~BCondBase() = default;

//             virtual void apply( Field<T, Dim, Mesh, Cell>& ) = 0;
//             virtual BCondBase<T, Dim, Mesh, Cell>* clone() const = 0;

            virtual void write(std::ostream&) const = 0;

                // Return component of Field element on which BC applies
            int getComponent() const { return component_m; }

            // Return face on which BC applies
            unsigned int getFace() const { return face_m; }

            // Returns whether or not this BC changes physical cells.
            bool changesPhysicalCells() const { return m_changePhysical; }

        protected:
            // Following are hooks for BC-by-Field-element-component support:
            // Component of Field elements (Vektor, e.g.) on which the BC applies:
            int component_m;

            // What face to apply the boundary condition to.
            unsigned int face_m;

            // True if this boundary condition changes physical cells.
            bool m_changePhysical;
        };


        template<typename T,
                 unsigned Dim,
                 class Mesh = UniformCartesian<double, Dim>,
                 class Cell = typename Mesh::DefaultCentering>
        class ExtrapolateFace : public BCondBase<T, Dim, Mesh, Cell>
        {
        public:
            // Constructor takes zero, one, or two int's specifying components of
            // multicomponent types like Vektor/Tenzor/Anti/SymTenzor this BC applies to.
            // Zero int's specified means apply to all components; one means apply to
            // component (i), and two means apply to component (i,j),
            using base_type = BCondBase<T, Dim, Mesh, Cell>;

            ExtrapolateFace(unsigned face,
                            T offset,
                            T slope,
                            int i = base_type::allComponents)
            : base_type(face, i)
            , offset_m(offset)
            , slope_m(slope)
            {}

            virtual ~ExtrapolateFace() = default;

        // Apply the boundary condition to a given Field.
//         virtual void apply( Field<T, Dim, Mesh, Cell>& );

//         // Make a copy of the concrete type.
//         virtual BCondBase<T, Dim, Mesh, Cell>* clone() const
//         {
//             return new ExtrapolateFace<T, Dim, Mesh, Cell>( *this );
//         }

        // Print out some information about the BC to a given stream.
        virtual void write(std::ostream&) const {};

        const T& getOffset() const { return offset_m; }
        const T& getSlope() const { return slope_m; }

        protected:
            T offset_m;
            T slope_m;
        };
    }


    template<typename T,
             unsigned Dim,
             class Mesh = UniformCartesian<double, Dim>,
             class Cell = typename Mesh::DefaultCentering>
    class ConstantFace : public detail::ExtrapolateFace<T, Dim, Mesh, Cell>
    {
    public:
        ConstantFace(unsigned int face, T constant)
        : detail::ExtrapolateFace<T, Dim, Mesh, Cell>(face, constant, 0)
        {}

        // Print out information about the BC to a stream.
        virtual void write(std::ostream& out) const;
    };
}


#include "Field/BcTypes.hpp"

#endif