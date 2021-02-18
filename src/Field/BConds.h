#ifndef IPPL_FIELD_BC_H
#define IPPL_FIELD_BC_H

#include "Field/BcTypes.h"

#include <array>
#include <iostream>
#include <memory>

namespace ippl {
    template<typename T, unsigned Dim, class Mesh, class Cell> class Field;

    //template<typename T, unsigned Dim, class Mesh, class Cell> class BConds;

    //template<typename T, unsigned Dim, class Mesh, class Cell>
    //std::ostream& operator<<(std::ostream&, const BConds<T, Dim, Mesh, Cell>&);
    template<unsigned Dim, class Mesh, class Cell> class BConds;

    template<unsigned Dim, class Mesh, class Cell>
    std::ostream& operator<<(std::ostream&, const BConds<Dim, Mesh, Cell>&);


    //template<typename T,
    //         unsigned Dim,
    //         class Mesh = UniformCartesian<double, Dim>,
    //         class Cell = typename Mesh::DefaultCentering>
    template<unsigned Dim,
             class Mesh = UniformCartesian<double, Dim>,
             class Cell = typename Mesh::DefaultCentering>
    class BConds //: public vmap<int, RefCountedP< BCondBase<T, Dim, Mesh, Cell> > >
    {
    public:
        using bc_type = detail::BCondBase<T, Dim, Mesh, Cell>;
        using container = std::array<std::shared_ptr<bc_type>, 2 * Dim>;
        using iterator = typename container::iterator;
        using const_iterator = typename container::const_iterator;

        void apply(Field<T, Dim, Mesh, Cell>& field);

        bool changesPhysicalCells() const;
        virtual void write(std::ostream&) const;

        const std::shared_ptr<bc_type>& operator[](const int& i) const noexcept {
            return bc_m[i];
        }

        std::shared_ptr<bc_type>& operator[](const int& i) noexcept {
            return bc_m[i];
        }

    private:
        container bc_m;
    };


    template<typename T, unsigned Dim, class Mesh, class Cell>
    inline std::ostream&
    operator<<(std::ostream& os, const BConds<T, Dim, Mesh, Cell>& bc)
    {
        bc.write(os);
        return os;
    }
}

// //////////////////////////////////////////////////////////////////////
// //BENI adds Periodic Boundary Conditions for Interpolations///////////
// //////////////////////////////////////////////////////////////////////
// template<class T,
//          unsigned D,
//          class M=UniformCartesian<D,double>,
//          class C= typename M::DefaultCentering>
// class InterpolationFace : public BCondBase<T, Dim, Mesh, Cell>
// {
// public:
//   // Constructor takes zero, one, or two int's specifying components of
//   // multicomponent types like Vektor/Tenzor/Anti/SymTenzor this BC applies to.
//   // Zero int's specified means apply to all components; one means apply to
//   // component (i), and two means apply to component (i,j),
//   typedef BCondBase<T, Dim, Mesh, Cell> BCondBaseTDMC;
//
//   InterpolationFace(unsigned f,
// 	       int i = BCondBaseTDMC::allComponents,
// 	       int j = BCondBaseTDMC::allComponents);
//
//   // Apply the boundary condition to a particular Field.
//   virtual void apply( Field<T, Dim, Mesh, Cell>& );
//
//   // Make a copy of the concrete type.
//   virtual BCondBase<T, Dim, Mesh, Cell>* clone() const
//   {
//     return new InterpolationFace<T, Dim, Mesh, Cell>( *this );
//   }
//
//   // Print out information about the BC to a stream.
//   virtual void write(std::ostream& out) const;
// };

//////////////////////////////////////////////////////////////////////

// template<class T, unsigned D,
//          class M=UniformCartesian<D,double>,
//          class C= typename M::DefaultCentering>
// class ParallelPeriodicFace : public PeriodicFace<T, Dim, Mesh, Cell>
// {
// public:
//
//   // Constructor takes zero, one, or two int's specifying components
//   // of multicomponent types like Vektor/Tenzor/AntiTenzor/SymTenzor
//   // this BC applies to.  Zero int's means apply to all components;
//   // one means apply to component (i), and two means apply to
//   // component (i,j),
//
//   typedef BCondBase<T, Dim, Mesh, Cell> Base_t;
//
//   ParallelPeriodicFace(unsigned f,
// 		       int i = Base_t::allComponents,
// 		       int j = Base_t::allComponents)
//     : PeriodicFace<T, Dim, Mesh, Cell>(f,i,j)
//   { }
//
//   // Apply the boundary condition to a particular Field.
//
//   virtual void apply( Field<T, Dim, Mesh, Cell>& );
//
//   // Make a copy of the concrete type.
//
//   virtual Base_t * clone() const
//   {
//     return new ParallelPeriodicFace<T, Dim, Mesh, Cell>( *this );
//   }
//
//   // Print out information about the BC to a stream.
//
//   virtual void write(std::ostream& out) const;
// };

//////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////
// BENI adds parallel Interpolation Face
//////////////////////////////////////////////////////////////////////

// template<class T, unsigned D,
//          class M=UniformCartesian<D,double>,
//          class C= typename M::DefaultCentering>
// class ParallelInterpolationFace : public InterpolationFace<T, Dim, Mesh, Cell>
// {
// public:
//
//   // Constructor takes zero, one, or two int's specifying components
//   // of multicomponent types like Vektor/Tenzor/AntiTenzor/SymTenzor
//   // this BC applies to.  Zero int's means apply to all components;
//   // one means apply to component (i), and two means apply to
//   // component (i,j),
//
//   typedef BCondBase<T, Dim, Mesh, Cell> Base_t;
//
//   ParallelInterpolationFace(unsigned f,
// 		       int i = Base_t::allComponents,
// 		       int j = Base_t::allComponents)
//     : InterpolationFace<T, Dim, Mesh, Cell>(f,i,j)
//   { }
//
//   // Apply the boundary condition to a particular Field.
//
//   virtual void apply( Field<T, Dim, Mesh, Cell>& );
//
//   // Make a copy of the concrete type.
//
//   virtual Base_t * clone() const
//   {
//     return new ParallelInterpolationFace<T, Dim, Mesh, Cell>( *this );
//   }
//
//   // Print out information about the BC to a stream.
//
//   virtual void write(std::ostream& out) const;
// };

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////

// TJW added 12/16/1997 as per Tecolote team's request: this one sets last
// physical element layer to zero for vert-centered elements/components. For
// cell-centered, doesn't need to do this because the zero point of an odd
// function is halfway between the last physical element and the first guard
// element.

// template<class T,
//          unsigned D,
//          class M=UniformCartesian<D,double>,
//          class C= typename M::DefaultCentering>
// class ExtrapolateAndZeroFace : public BCondBase<T, Dim, Mesh, Cell>
// {
// public:
//   // Constructor takes zero, one, or two int's specifying components of
//   // multicomponent types like Vektor/Tenzor/Anti/SymTenzor this BC applies to.
//   // Zero int's specified means apply to all components; one means apply to
//   // component (i), and two means apply to component (i,j),
//   typedef BCondBase<T, Dim, Mesh, Cell> BCondBaseTDMC;
//   ExtrapolateAndZeroFace(unsigned f, T o, T s,
// 		  int i = BCondBaseTDMC::allComponents,
// 		  int j = BCondBaseTDMC::allComponents);
//
//   // Apply the boundary condition to a given Field.
//   virtual void apply( Field<T, Dim, Mesh, Cell>& );
//
//   // Make a copy of the concrete type.
//   virtual BCondBase<T, Dim, Mesh, Cell>* clone() const
//   {
//     return new ExtrapolateAndZeroFace<T, Dim, Mesh, Cell>( *this );
//   }
//
//   // Print out some information about the BC to a given stream.
//   virtual void write(std::ostream&) const;
//
//   const T& getOffset() const { return Offset; }
//   const T& getSlope() const { return Slope; }
//
// protected:
//   T Offset, Slope;
// };

//////////////////////////////////////////////////////////////////////

// template<class T,
//          unsigned D,
//          class M=UniformCartesian<D,double>,
//          class C= typename M::DefaultCentering>
// class ZeroFace : public ExtrapolateFace<T, Dim, Mesh, Cell>
// {
// public:
//   typedef BCondBase<T, Dim, Mesh, Cell> BCondBaseTDMC;
//   ZeroFace(unsigned f,
// 	   int i = BCondBaseTDMC::allComponents,
// 	   int j = BCondBaseTDMC::allComponents)
//     : ExtrapolateFace<T, Dim, Mesh, Cell>(f,0,0,i,j) {}
//
//   // Print out information about the BC to a stream.
//   virtual void write(std::ostream& out) const;
// };

//////////////////////////////////////////////////////////////////////

// TJW added 1/25/1998 as per Blanca's (Don Marshal's, for the Lagrangian
// code) request: this one sets last physical element layer to zero for
// vert-centered elements/components. For cell-centered, doesn't need to do
// this because the zero point of an odd function is halfway between the last
// physical element and the first guard element.

// template<class T,
//          unsigned D,
//          class M=UniformCartesian<D,double>,
//          class C= typename M::DefaultCentering>
// class ZeroGuardsAndZeroFace : public ExtrapolateAndZeroFace<T, Dim, Mesh, Cell>
// {
// public:
//   typedef BCondBase<T, Dim, Mesh, Cell> BCondBaseTDMC;
//   ZeroGuardsAndZeroFace(unsigned f,
// 			int i = BCondBaseTDMC::allComponents,
// 			int j = BCondBaseTDMC::allComponents)
//     : ExtrapolateAndZeroFace<T, Dim, Mesh, Cell>(f,0,0,i,j) {}
//
//   // Print out information about the BC to a stream.
//   virtual void write(std::ostream& out) const;
// };


//////////////////////////////////////////////////////////////////////

// ----------------------------------------------------------------------------
// TJW added 1/26/1998 as per Blanca's (Jerry Brock's, for the tracer particle
// code) request: this one takes the values of the last two physical elements,
// and linearly extrapolates from the line through them out to all the guard
// elements. This is independent of centering. The intended use is for filling
// global guard layers of a Field of Vektors holding the mesh node position
// values, for which this does the right thing at hi and lo faces (exactly
// right for uniform cartesian meshes, and a reasonable thing to do for
// nonuniform cartesian meshes).
// 
// Had to depart from the paradigm of the other boundary condition types to
// implement LinearExtrapolateFace. Couldn't use the same single
// LinearExtrapolateFace class for both whole-Field-element and componentwise
// functions, because I couldn't figure out how to implement it using PETE and
// applicative templates.  Instead, created LinearExtrapolateFace as
// whole-element-only (no component specification allowed) and separate
// ComponentLinearExtrapolateFace for componentwise, disallowing via runtime
// error the allComponents specification allowed in all the other BC
// types. --Tim Williams 1/26/1999
// ----------------------------------------------------------------------------

// template<class T,
//          unsigned D,
//          class M=UniformCartesian<D,double>,
//          class C= typename M::DefaultCentering>
// class LinearExtrapolateFace : public BCondBase<T, Dim, Mesh, Cell>
// {
// public:
//   // Constructor takes zero, one, or two int's specifying components of
//   // multicomponent types like Vektor/Tenzor/Anti/SymTenzor this BC applies to.
//   // Zero int's specified means apply to all components; one means apply to
//   // component (i), and two means apply to component (i,j),
//   typedef BCondBase<T, Dim, Mesh, Cell> BCondBaseTDMC;
//   LinearExtrapolateFace(unsigned f) :
//     BCondBase<T, Dim, Mesh, Cell>(f) {}
//
//   // Apply the boundary condition to a given Field.
//   virtual void apply( Field<T, Dim, Mesh, Cell> &A);
//
//   // Make a copy of the concrete type.
//   virtual BCondBase<T, Dim, Mesh, Cell>* clone() const
//   {
//     return new LinearExtrapolateFace<T, Dim, Mesh, Cell>( *this );
//   }
//
//   // Print out some information about the BC to a given stream.
//   virtual void write(std::ostream&) const;
//
// };


//////////////////////////////////////////////////////////////////////

//
// Define global streaming functions that just call the
// write function for each of the above base classes.
//

#include "Field/BConds.hpp"

#endif
