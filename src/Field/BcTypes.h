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
            // Constructor takes:
            // face: the face to apply the boundary condition on.
            // i : what component of T to apply the boundary condition to.
            // The components default to setting all components.
            BCondBase(unsigned int face);

            virtual ~BCondBase() = default;

//             virtual void apply( Field<T, Dim, Mesh, Cell>& ) = 0;

            virtual void write(std::ostream&) const = 0;

            // Return face on which BC applies
            unsigned int getFace() const { return face_m; }

            // Returns whether or not this BC changes physical cells.
            bool changesPhysicalCells() const { return changePhysical_m; }

        protected:
            // What face to apply the boundary condition to.
            unsigned int face_m;

            // True if this boundary condition changes physical cells.
            bool changePhysical_m;
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
                            T slope)
            : base_type(face)
            , offset_m(offset)
            , slope_m(slope)
            {}

            virtual ~ExtrapolateFace() = default;

//         virtual void apply( Field<T, Dim, Mesh, Cell>& );

            virtual void write(std::ostream&) const = 0;

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
    class NoBcFace : public detail::BCondBase<T, Dim, Mesh, Cell>
    {
        public:
            NoBcFace(int face) : detail::BCondBase<T, Dim, Mesh, Cell>(face) {}

    //     virtual void apply( Field<T, Dim, Mesh, Cell>& ) {}

            virtual void write(std::ostream& out) const;
    };


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

        virtual void write(std::ostream& out) const;
    };


    template<typename T,
             unsigned Dim,
             class Mesh = UniformCartesian<double, Dim>,
             class Cell = typename Mesh::DefaultCentering>
    class ZeroFace : public ConstantFace<T, Dim, Mesh, Cell>
    {
    public:
        ZeroFace(unsigned face)
        : ConstantFace<T, Dim, Mesh, Cell>(face, 0.0)
        {}

        virtual void write(std::ostream& out) const;
    };


    template<typename T,
             unsigned Dim,
             class Mesh = UniformCartesian<double, Dim>,
             class Cell = typename Mesh::DefaultCentering>
    class PeriodicFace : public detail::BCondBase<T, Dim, Mesh, Cell>
    {
    public:
        PeriodicFace(unsigned face)
        : detail::BCondBase<T, Dim, Mesh, Cell>(face)
        { }

//         virtual void apply( Field<T, Dim, Mesh, Cell>& );

        virtual void write(std::ostream& out) const;
    };
}


#include "Field/BcTypes.hpp"

#endif