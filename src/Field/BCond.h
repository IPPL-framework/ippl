// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 ***************************************************************************/

#ifndef BCOND_H
#define BCOND_H

#include "Utility/IpplInfo.h"
#include "Utility/RefCounted.h"
#include "Utility/vmap.h"

#include <iostream>
#include <complex>

// forward declarations
template <unsigned D> class NDIndex;
template <class T, unsigned D> class Vektor;
template <class T, unsigned D> class Tenzor;
template <class T, unsigned D> class SymTenzor;
template <class T, unsigned D> class AntiSymTenzor;
template<unsigned D, class T> class UniformCartesian;
template<class T, unsigned D> class LField;
template<class T, unsigned D> class BareField;
template<class T, unsigned D, class M, class C> class Field;
template <class T, unsigned D, class M, class C> class BCondBase;
template <class T, unsigned D, class M, class C>
std::ostream& operator<<(std::ostream&, const BCondBase<T,D,M,C>&);
template <class T, unsigned D, class M, class C> class BConds;
template <class T, unsigned D, class M, class C>
std::ostream& operator<<(std::ostream&, const BConds<T,D,M,C>&);

//////////////////////////////////////////////////////////////////////

//
// Traits used by the single-component version of the applicative templates.
// General case: this covers intrinsic types like double, bool automatically:
//

template<class T>
struct ApplyToComponentType 
{
  typedef T type;
};

//
// Specializations for multicomponent IPPL types;
//
template<class T,unsigned D>
struct ApplyToComponentType< Vektor<T,D> > 
{
  typedef T type;
};

template<class T,unsigned D>
struct ApplyToComponentType< Tenzor<T,D> > 
{
  typedef T type;
};

template<class T,unsigned D>
struct ApplyToComponentType< SymTenzor<T,D> > 
{
  typedef T type;
};

template<class T,unsigned D>
struct ApplyToComponentType< AntiSymTenzor<T,D> > 
{
  typedef T type;
};

// Helper classes for getting info about number of indices into 
// BCond-class ctor functions.
// Define tag types (like iterator tags in stl):

class scalar_tag
{
};

class vektor_tag
{
};

class tenzor_tag
{
};

class symtenzor_tag
{
};

class antisymtenzor_tag
{
};

// Implement tag types for intrinsic types:
inline scalar_tag get_tag(std::complex<double>) { return scalar_tag(); }
inline scalar_tag get_tag(double)   { return scalar_tag(); }
inline scalar_tag get_tag(float)    { return scalar_tag(); }
inline scalar_tag get_tag(int)      { return scalar_tag(); }
inline scalar_tag get_tag(bool)     { return scalar_tag(); }
inline scalar_tag get_tag(short)    { return scalar_tag(); }

// Tag for Vektor types:
template<class T, unsigned D>
inline vektor_tag 
get_tag(Vektor<T,D>) { return vektor_tag(); }

// Tag for Tenzor types:
template<class T, unsigned D>
inline tenzor_tag 
get_tag(Tenzor<T,D>) { return tenzor_tag(); }

// Tag for AntiSymTenzor types
template<class T, unsigned D>
inline antisymtenzor_tag 
get_tag(AntiSymTenzor<T,D>) { return antisymtenzor_tag(); }

// Tag for SymTenzor types
template<class T, unsigned D>
inline symtenzor_tag 
get_tag(SymTenzor<T,D>) { return symtenzor_tag(); }

// Functions which return an enum value indicating scalar, vector, tensor,
// or anti/symtensor type; used in constructors for PeriodicFace, etc., to
// determine how to turn two 1D component indices into a single index value
// for pointer offsetting into the Tenzor/Anti/SymTenzor object:
enum TensorOrder_e { IPPL_SCALAR=0, IPPL_VECTOR=1, IPPL_TENSOR=2, 
		     IPPL_SYMTENSOR=3, IPPL_ANTISYMTENSOR=4 } ;
inline TensorOrder_e getTensorOrder(const scalar_tag& )
{return IPPL_SCALAR;}
inline TensorOrder_e getTensorOrder(const vektor_tag& )
{return IPPL_VECTOR;}
inline TensorOrder_e getTensorOrder(const tenzor_tag& )
{return IPPL_TENSOR;}
inline TensorOrder_e getTensorOrder(const antisymtenzor_tag& )
{return IPPL_ANTISYMTENSOR;}
inline TensorOrder_e getTensorOrder(const symtenzor_tag& )
{return IPPL_SYMTENSOR;}

//////////////////////////////////////////////////////////////////////

template<class T, unsigned D, class M, class C>
class BCondBase : public RefCounted
{
public:

  // Special value designating application to all components of elements:
  static int allComponents;

  // Constructor takes:
  // face: the face to apply the boundary condition on.
  // i,j : what component of T to apply the boundary condition to.
  // The components default to setting all components.
  BCondBase(unsigned int face,
	    int i = allComponents,
	    int j = allComponents);
  virtual ~BCondBase() { }

  virtual void apply( Field<T,D,M,C>& ) = 0;
  virtual BCondBase<T,D,M,C>* clone() const = 0;

  virtual void write(std::ostream&) const;

  // Return component of Field element on which BC applies
  int getComponent() const { return m_component; }

  // Return face on which BC applies
  unsigned int getFace() const { return m_face; }

  // Returns whether or not this BC changes physical cells.
  bool changesPhysicalCells() const { return m_changePhysical; }

protected:

  // Following are hooks for BC-by-Field-element-component support:
  // Component of Field elements (Vektor, e.g.) on which the BC applies:
  int m_component;
  
  // What face to apply the boundary condition to.
  unsigned int m_face;

  // True if this boundary condition changes physical cells.
  bool m_changePhysical;
};

//////////////////////////////////////////////////////////////////////

template<
  class T,
  unsigned D,
  class M=UniformCartesian<D,double>,
  class C=typename M::DefaultCentering>
class BConds
  : public vmap<int, RefCountedP< BCondBase<T,D,M,C> > >
{
public: 
  typedef typename vmap<int, RefCountedP <BCondBase<T,D,M,C> > >::iterator 
    iterator; 
  typedef typename vmap<int, RefCountedP <BCondBase<T,D,M,C> > >::const_iterator 
    const_iterator; 
  void apply( Field<T,D,M,C>& a );
  bool changesPhysicalCells() const;
  virtual void write(std::ostream&) const;
};

//////////////////////////////////////////////////////////////////////

// TJW: so far, componentwise specification of BCondNoAction not possible

template<class T,
         unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class BCondNoAction : public BCondBase<T,D,M,C>
{
public:
  BCondNoAction(int face) : BCondBase<T,D,M,C>(face) {}

  virtual void apply( Field<T,D,M,C>& ) {}
  BCondBase<T,D,M,C>* clone() const
  {
    return new BCondNoAction<T,D,M,C>( *this );
  }

  // Print out information about the BC to a stream.
  virtual void write(std::ostream& out) const;
};

//////////////////////////////////////////////////////////////////////

template<class T,
         unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class PeriodicFace : public BCondBase<T,D,M,C>
{
public:
  // Constructor takes zero, one, or two int's specifying components of 
  // multicomponent types like Vektor/Tenzor/Anti/SymTenzor this BC applies to.
  // Zero int's specified means apply to all components; one means apply to
  // component (i), and two means apply to component (i,j),
  typedef BCondBase<T,D,M,C> BCondBaseTDMC;

  PeriodicFace(unsigned f, 
	       int i = BCondBaseTDMC::allComponents,
	       int j = BCondBaseTDMC::allComponents);

  // Apply the boundary condition to a particular Field.
  virtual void apply( Field<T,D,M,C>& );

  // Make a copy of the concrete type.
  virtual BCondBase<T,D,M,C>* clone() const
  {
    return new PeriodicFace<T,D,M,C>( *this );
  }

  // Print out information about the BC to a stream.
  virtual void write(std::ostream& out) const;
};



//////////////////////////////////////////////////////////////////////
//BENI adds Periodic Boundary Conditions for Interpolations///////////
//////////////////////////////////////////////////////////////////////
template<class T,
         unsigned D,
         class M=UniformCartesian<D,double>,
         class C=typename M::DefaultCentering>
class InterpolationFace : public BCondBase<T,D,M,C>
{
public:
  // Constructor takes zero, one, or two int's specifying components of
  // multicomponent types like Vektor/Tenzor/Anti/SymTenzor this BC applies to.
  // Zero int's specified means apply to all components; one means apply to
  // component (i), and two means apply to component (i,j),
  typedef BCondBase<T,D,M,C> BCondBaseTDMC;

  InterpolationFace(unsigned f,
	       int i = BCondBaseTDMC::allComponents,
	       int j = BCondBaseTDMC::allComponents);

  // Apply the boundary condition to a particular Field.
  virtual void apply( Field<T,D,M,C>& );

  // Make a copy of the concrete type.
  virtual BCondBase<T,D,M,C>* clone() const
  {
    return new InterpolationFace<T,D,M,C>( *this );
  }

  // Print out information about the BC to a stream.
  virtual void write(std::ostream& out) const;
};

//////////////////////////////////////////////////////////////////////

template<class T, unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class ParallelPeriodicFace : public PeriodicFace<T,D,M,C>
{
public:

  // Constructor takes zero, one, or two int's specifying components
  // of multicomponent types like Vektor/Tenzor/AntiTenzor/SymTenzor
  // this BC applies to.  Zero int's means apply to all components;
  // one means apply to component (i), and two means apply to
  // component (i,j),

  typedef BCondBase<T,D,M,C> Base_t;

  ParallelPeriodicFace(unsigned f, 
		       int i = Base_t::allComponents,
		       int j = Base_t::allComponents)
    : PeriodicFace<T,D,M,C>(f,i,j) 
  { }

  // Apply the boundary condition to a particular Field.

  virtual void apply( Field<T,D,M,C>& );

  // Make a copy of the concrete type.

  virtual Base_t * clone() const
  {
    return new ParallelPeriodicFace<T,D,M,C>( *this );
  }

  // Print out information about the BC to a stream.

  virtual void write(std::ostream& out) const;
};

//////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////
// BENI adds parallel Interpolation Face
//////////////////////////////////////////////////////////////////////

template<class T, unsigned D,
         class M=UniformCartesian<D,double>,
         class C=typename M::DefaultCentering>
class ParallelInterpolationFace : public InterpolationFace<T,D,M,C>
{
public:

  // Constructor takes zero, one, or two int's specifying components
  // of multicomponent types like Vektor/Tenzor/AntiTenzor/SymTenzor
  // this BC applies to.  Zero int's means apply to all components;
  // one means apply to component (i), and two means apply to
  // component (i,j),

  typedef BCondBase<T,D,M,C> Base_t;

  ParallelInterpolationFace(unsigned f,
		       int i = Base_t::allComponents,
		       int j = Base_t::allComponents)
    : InterpolationFace<T,D,M,C>(f,i,j)
  { }

  // Apply the boundary condition to a particular Field.

  virtual void apply( Field<T,D,M,C>& );

  // Make a copy of the concrete type.

  virtual Base_t * clone() const
  {
    return new ParallelInterpolationFace<T,D,M,C>( *this );
  }

  // Print out information about the BC to a stream.

  virtual void write(std::ostream& out) const;
};

//////////////////////////////////////////////////////////////////////////


template<class T,
         unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class ExtrapolateFace : public BCondBase<T,D,M,C>
{
public:
  // Constructor takes zero, one, or two int's specifying components of 
  // multicomponent types like Vektor/Tenzor/Anti/SymTenzor this BC applies to.
  // Zero int's specified means apply to all components; one means apply to
  // component (i), and two means apply to component (i,j),
  typedef BCondBase<T,D,M,C> BCondBaseTDMC;
  ExtrapolateFace(unsigned f, T o, T s, 
		  int i = BCondBaseTDMC::allComponents,
		  int j = BCondBaseTDMC::allComponents);

  // Apply the boundary condition to a given Field.
  virtual void apply( Field<T,D,M,C>& );

  // Make a copy of the concrete type.
  virtual BCondBase<T,D,M,C>* clone() const
  {
    return new ExtrapolateFace<T,D,M,C>( *this );
  }

  // Print out some information about the BC to a given stream.
  virtual void write(std::ostream&) const;

  const T& getOffset() const { return Offset; }
  const T& getSlope() const { return Slope; }

protected:
  T Offset, Slope;
};


//////////////////////////////////////////////////////////////////////

// TJW added 12/16/1997 as per Tecolote team's request: this one sets last
// physical element layer to zero for vert-centered elements/components. For
// cell-centered, doesn't need to do this because the zero point of an odd
// function is halfway between the last physical element and the first guard
// element.

template<class T,
         unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class ExtrapolateAndZeroFace : public BCondBase<T,D,M,C>
{
public:
  // Constructor takes zero, one, or two int's specifying components of 
  // multicomponent types like Vektor/Tenzor/Anti/SymTenzor this BC applies to.
  // Zero int's specified means apply to all components; one means apply to
  // component (i), and two means apply to component (i,j),
  typedef BCondBase<T,D,M,C> BCondBaseTDMC;
  ExtrapolateAndZeroFace(unsigned f, T o, T s, 
		  int i = BCondBaseTDMC::allComponents,
		  int j = BCondBaseTDMC::allComponents);

  // Apply the boundary condition to a given Field.
  virtual void apply( Field<T,D,M,C>& );

  // Make a copy of the concrete type.
  virtual BCondBase<T,D,M,C>* clone() const
  {
    return new ExtrapolateAndZeroFace<T,D,M,C>( *this );
  }

  // Print out some information about the BC to a given stream.
  virtual void write(std::ostream&) const;

  const T& getOffset() const { return Offset; }
  const T& getSlope() const { return Slope; }

protected:
  T Offset, Slope;
};


//////////////////////////////////////////////////////////////////////

template<class T,
         unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class PosReflectFace : public ExtrapolateFace<T,D,M,C>
{
public: 
  typedef BCondBase<T,D,M,C> BCondBaseTDMC;
  PosReflectFace(unsigned f,
		 int i = BCondBaseTDMC::allComponents,
		 int j = BCondBaseTDMC::allComponents)
    : ExtrapolateFace<T,D,M,C>(f,0,1,i,j) {}

  // Print out information about the BC to a stream.
  virtual void write(std::ostream& out) const;
};

//////////////////////////////////////////////////////////////////////

template<class T,
         unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class NegReflectFace : public ExtrapolateFace<T,D,M,C>
{
public: 
  typedef BCondBase<T,D,M,C> BCondBaseTDMC;
  NegReflectFace(unsigned f,
		 int i = BCondBaseTDMC::allComponents,
		 int j = BCondBaseTDMC::allComponents)
    : ExtrapolateFace<T,D,M,C>(f,0,-1,i,j) {}

  // Print out information about the BC to a stream.
  virtual void write(std::ostream& out) const;
};

//////////////////////////////////////////////////////////////////////

// TJW added 12/16/1997 as per Tecolote team's request: this one sets last
// physical element layer to zero for vert-centered elements/components. For
// cell-centered, doesn't need to do this because the zero point of an odd
// function is halfway between the last physical element and the first guard
// element.

template<class T,
         unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class NegReflectAndZeroFace : public ExtrapolateAndZeroFace<T,D,M,C>
{
public: 
  typedef BCondBase<T,D,M,C> BCondBaseTDMC;
  NegReflectAndZeroFace(unsigned f,
			int i = BCondBaseTDMC::allComponents,
			int j = BCondBaseTDMC::allComponents)
    : ExtrapolateAndZeroFace<T,D,M,C>(f,0,-1,i,j) {}

  // Print out information about the BC to a stream.
  virtual void write(std::ostream& out) const;
};

//////////////////////////////////////////////////////////////////////

template<class T,
         unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class ConstantFace : public ExtrapolateFace<T,D,M,C>
{
public: 
  typedef BCondBase<T,D,M,C> BCondBaseTDMC;
  ConstantFace(unsigned f, T c,
	       int i = BCondBaseTDMC::allComponents,
	       int j = BCondBaseTDMC::allComponents)
    : ExtrapolateFace<T,D,M,C>(f,c,0,i,j) {}

  // Print out information about the BC to a stream.
  virtual void write(std::ostream& out) const;
};

//////////////////////////////////////////////////////////////////////

template<class T,
         unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class ZeroFace : public ExtrapolateFace<T,D,M,C>
{
public: 
  typedef BCondBase<T,D,M,C> BCondBaseTDMC;
  ZeroFace(unsigned f,
	   int i = BCondBaseTDMC::allComponents,
	   int j = BCondBaseTDMC::allComponents)
    : ExtrapolateFace<T,D,M,C>(f,0,0,i,j) {}

  // Print out information about the BC to a stream.
  virtual void write(std::ostream& out) const;
};

//////////////////////////////////////////////////////////////////////

// TJW added 1/25/1998 as per Blanca's (Don Marshal's, for the Lagrangian
// code) request: this one sets last physical element layer to zero for
// vert-centered elements/components. For cell-centered, doesn't need to do
// this because the zero point of an odd function is halfway between the last
// physical element and the first guard element.

template<class T,
         unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class ZeroGuardsAndZeroFace : public ExtrapolateAndZeroFace<T,D,M,C>
{
public: 
  typedef BCondBase<T,D,M,C> BCondBaseTDMC;
  ZeroGuardsAndZeroFace(unsigned f,
			int i = BCondBaseTDMC::allComponents,
			int j = BCondBaseTDMC::allComponents)
    : ExtrapolateAndZeroFace<T,D,M,C>(f,0,0,i,j) {}

  // Print out information about the BC to a stream.
  virtual void write(std::ostream& out) const;
};


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

template<class T,
         unsigned D, 
         class M=UniformCartesian<D,double>, 
         class C=typename M::DefaultCentering>
class LinearExtrapolateFace : public BCondBase<T,D,M,C>
{
public:
  // Constructor takes zero, one, or two int's specifying components of 
  // multicomponent types like Vektor/Tenzor/Anti/SymTenzor this BC applies to.
  // Zero int's specified means apply to all components; one means apply to
  // component (i), and two means apply to component (i,j),
  typedef BCondBase<T,D,M,C> BCondBaseTDMC;
  LinearExtrapolateFace(unsigned f) :
    BCondBase<T,D,M,C>(f) {}

  // Apply the boundary condition to a given Field.
  virtual void apply( Field<T,D,M,C> &A);

  // Make a copy of the concrete type.
  virtual BCondBase<T,D,M,C>* clone() const
  {
    return new LinearExtrapolateFace<T,D,M,C>( *this );
  }

  // Print out some information about the BC to a given stream.
  virtual void write(std::ostream&) const;

};


//////////////////////////////////////////////////////////////////////

//
// Define global streaming functions that just call the
// write function for each of the above base classes.
//

template<class T, unsigned D, class M, class C >
inline std::ostream&
operator<<(std::ostream& o, const BCondBase<T,D,M,C>& bc)
{
  bc.write(o);
  return o;
}


template<class T, unsigned D, class M, class C >
inline std::ostream&
operator<<(std::ostream& o, const BConds<T,D,M,C>& bc)
{
  bc.write(o);
  return o;
}

//////////////////////////////////////////////////////////////////////

#include "Field/BCond.hpp"

#endif