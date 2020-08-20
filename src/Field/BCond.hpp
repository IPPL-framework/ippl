// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

// include files
#include "Field/BCond.h"
#include "Field/BareField.h"
#include "Index/NDIndex.h"
#include "Index/Index.h"
#include "Field/GuardCellSizes.h"
#include "Field/BrickIterator.h"
#include "Field/BrickExpression.h"
#include "Meshes/Centering.h"
#include "Meshes/CartesianCentering.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"
#include "AppTypes/AppTypeTraits.h"


#include <iostream>
#include <typeinfo>
#include <vector>

//////////////////////////////////////////////////////////////////////

template<class T, unsigned D, class M, class C>
int BCondBase<T,D,M,C>::allComponents = -9999;

//////////////////////////////////////////////////////////////////////

// Use this macro to specialize PETE_apply functions for component-wise
// operators and built-in types and print an error message.

#define COMPONENT_APPLY_BUILTIN(OP,T)                                       \
inline void PETE_apply(const OP<T>&, T&, const T&)                          \
{                                                                           \
  ERRORMSG("Component boundary condition on a scalar (T)." << endl);        \
  Ippl::abort();                                                           \
}


/*

  Constructor for BCondBase<T,D,M,C>
  Records the face, and figures out what component to remember.

 */

template<class T, unsigned int D, class M, class C>
BCondBase<T,D,M,C>::BCondBase(unsigned int face, int i, int j)
: m_face(face), m_changePhysical(false)
{
  // Must figure out if two, one, or no component indices are specified
  // for which this BC is to apply; of none are specified, it applies to
  // all components of the elements (of type T).
  if (j != BCondBase<T,D,M,C>::allComponents) {
    if (i == BCondBase<T,D,M,C>::allComponents)
      ERRORMSG("BCondBase(): component 2 specified, component 1 not."
	       << endl);
    // For two specified component indices, must turn the two integer component
    // index values into a single int value for Component, which is used in
    // pointer offsets in applicative templates elsewhere. How to do this
    // depends on what kind of two-index multicomponent object T is. Implement
    // only for Tenzor, AntiSymTenzor, and SymTenzor (for now, at least):
    if (getTensorOrder(get_tag(T())) == IPPL_TENSOR) {
      m_component = i + j*D;
    } else if (getTensorOrder(get_tag(T())) == IPPL_SYMTENSOR) {
      int lo = i < j ? i : j;
      int hi = i > j ? i : j;
      m_component = ((hi+1)*hi/2) + lo;
    } else if (getTensorOrder(get_tag(T())) == IPPL_ANTISYMTENSOR) {
      PAssert_GT(i, j);
      m_component = ((i-1)*i/2) + j;
    } else {
      ERRORMSG(
        "BCondBase(): something other than [Sym,AntiSym]Tenzor specified"
	<< " two component indices; not implemented." << endl);
    }

  } else {
    // For only one specified component index (including the default case of
    // BCondBase::allComponents meaning apply to all components of T, just
    // assign the Component value for use in pointer offsets into
    // single-component-index types in applicative templates elsewhere:
    m_component = i;
  }
}

//////////////////////////////////////////////////////////////////////

/*

  BCondBase::write(ostream&)
  Print out information about the BCondBase to an ostream.
  This is called by its subclasses, which is why
  it calls typeid(*this) to print out the class name.

 */

template<class T, unsigned int D, class M, class C>
void BCondBase<T,D,M,C>::write(std::ostream& out) const
{
  out << "BCondBase" << ", Face=" << m_face;
}

template<class T, unsigned int D, class M, class C>
void PeriodicFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "PeriodicFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

//BENI adds Interpolation face BC
template<class T, unsigned int D, class M, class C>
void InterpolationFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "InterpolationFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

//BENI adds ParallelInterpolation face BC
template<class T, unsigned int D, class M, class C>
void ParallelInterpolationFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "ParallelInterpolationFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

template<class T, unsigned int D, class M, class C>
void ParallelPeriodicFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "ParallelPeriodicFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

template<class T, unsigned int D, class M, class C>
void NegReflectFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "NegReflectFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

template<class T, unsigned int D, class M, class C>
void NegReflectAndZeroFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "NegReflectAndZeroFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

template<class T, unsigned int D, class M, class C>
void PosReflectFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "PosReflectFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

template<class T, unsigned int D, class M, class C>
void ZeroFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "ZeroFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

template<class T, unsigned int D, class M, class C>
void ZeroGuardsAndZeroFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "ZeroGuardsAndZeroFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

template<class T, unsigned int D, class M, class C>
void ConstantFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "ConstantFace"
      << ", Face=" << BCondBase<T,D,M,C>::m_face
      << ", Constant=" << this->Offset
      << std::endl;
}

template<class T, unsigned int D, class M, class C>
void EurekaFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "EurekaFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

template<class T, unsigned int D, class M, class C>
void FunctionFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "FunctionFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

template<class T, unsigned int D, class M, class C>
void ComponentFunctionFace<T,D,M,C>::write(std::ostream& out) const
{
  out << "ComponentFunctionFace" << ", Face=" << BCondBase<T,D,M,C>::m_face;
}

template<class T, unsigned D, class M, class C>
void
ExtrapolateFace<T,D,M,C>::write(std::ostream& o) const
{


  o << "ExtrapolateFace, Face=" << BCondBase<T,D,M,C>::m_face
    << ", Offset=" << Offset << ", Slope=" << Slope;
}

template<class T, unsigned D, class M, class C>
void
ExtrapolateAndZeroFace<T,D,M,C>::write(std::ostream& o) const
{


  o << "ExtrapolateAndZeroFace, Face=" << BCondBase<T,D,M,C>::m_face
    << ", Offset=" << Offset << ", Slope=" << Slope;
}

template<class T, unsigned D, class M, class C>
void
LinearExtrapolateFace<T,D,M,C>::write(std::ostream& o) const
{


  o << "LinearExtrapolateFace, Face=" << BCondBase<T,D,M,C>::m_face;
}

template<class T, unsigned D, class M, class C>
void
ComponentLinearExtrapolateFace<T,D,M,C>::write(std::ostream& o) const
{

  o << "ComponentLinearExtrapolateFace, Face=" << BCondBase<T,D,M,C>::m_face;
}

//////////////////////////////////////////////////////////////////////

template<class T, unsigned D, class M, class C>
void
BConds<T,D,M,C>::write(std::ostream& o) const
{



  o << "BConds:(" << std::endl;
  const_iterator p=this->begin();
  while (p!=this->end())
    {
      (*p).second->write(o);
      ++p;
      if (p!=this->end())
        o << " , " << std::endl;
      else
          o << std::endl << ")" << std::endl << std::endl;
    }
}

//////////////////////////////////////////////////////////////////////

template<class T, unsigned D, class M, class C>
void
BConds<T,D,M,C>::apply( Field<T,D,M,C>& a )
{


  for (iterator p=this->begin(); p!=this->end(); ++p)
    (*p).second->apply(a);
}

template<class T, unsigned D, class M, class C>
bool
BConds<T,D,M,C>::changesPhysicalCells() const
{
  for (const_iterator p=this->begin(); p!=this->end(); ++p)
    if ((*p).second->changesPhysicalCells())
      return true;
  return false;
}

//=============================================================================
// Constructors for PeriodicFace, ExtrapolateFace, FunctionFace classes
// and, now, ComponentFunctionFace
//=============================================================================

template<class T, unsigned D, class M, class C>
PeriodicFace<T,D,M,C>::PeriodicFace(unsigned f, int i, int j)
  : BCondBase<T,D,M,C>(f,i,j)
{
	//std::cout << "(1) Constructor periodic face called" << std::endl;


}

//BENI adds Interpolation face BC
template<class T, unsigned D, class M, class C>
InterpolationFace<T,D,M,C>::InterpolationFace(unsigned f, int i, int j)
  : BCondBase<T,D,M,C>(f,i,j)
{


}

template<class T, unsigned D, class M, class C>
ExtrapolateFace<T,D,M,C>::ExtrapolateFace(unsigned f, T o, T s, int i, int j)
  : BCondBase<T,D,M,C>(f,i,j), Offset(o), Slope(s)
{


}

template<class T, unsigned D, class M, class C>
ExtrapolateAndZeroFace<T,D,M,C>::
ExtrapolateAndZeroFace(unsigned f, T o, T s, int i, int j)
  : BCondBase<T,D,M,C>(f,i,j), Offset(o), Slope(s)
{

  BCondBase<T,D,M,C>::m_changePhysical = true;
}

template<class T, unsigned D, class M, class C>
FunctionFace<T,D,M,C>::
FunctionFace(T (*func)(const T&), unsigned face)
  : BCondBase<T,D,M,C>(face), Func(func)
{


}

template<class T, unsigned D, class M, class C>
ComponentFunctionFace<T,D,M,C>::
ComponentFunctionFace(typename ApplyToComponentType<T>::type
		      (*func)( typename ApplyToComponentType<T>::type),
		      unsigned face, int i, int j)
  : BCondBase<T,D,M,C>(face,i,j), Func(func)
{

  // Disallow specification of all components (default, unfortunately):
  if ( (j == BCondBase<T,D,M,C>::allComponents) &&
       (i == BCondBase<T,D,M,C>::allComponents) )
    ERRORMSG("ComponentFunctionFace(): allComponents specified; not allowed; "
	     << "use FunctionFace class instead." << endl);
}


//////////////////////////////////////////////////////////////////////

// Applicative templates for PeriodicFace:

// Standard, for applying to all components of elemental type:
// (N.B.: could just use OpAssign, but put this in for clarity and consistency
// with other appliciative templates in this module.)
template<class T>
struct OpPeriodic
{
};
template<class T>
inline void PETE_apply(const OpPeriodic<T>& /*e*/, T& a, const T& b) {a = b; }

// Special, for applying to single component of multicomponent elemental type:
template<class T>
struct OpPeriodicComponent
{
  OpPeriodicComponent(int c) : Component(c) {}
  int Component;
};

template<class T>
inline void PETE_apply(const OpPeriodicComponent<T>& e, T& a, const T& b)
{ a[e.Component] = b[e.Component]; }

// Following specializations are necessary because of the runtime branches in
// code which unfortunately force instantiation of OpPeriodicComponent
// instances for non-multicomponent types like {char,double,...}.
// Note: if user uses non-multicomponent (no operator[]) types of his own,
// he'll get a compile error. See comments regarding similar specializations
// for OpExtrapolateComponent for a more details.
COMPONENT_APPLY_BUILTIN(OpPeriodicComponent,char)
COMPONENT_APPLY_BUILTIN(OpPeriodicComponent,bool)
COMPONENT_APPLY_BUILTIN(OpPeriodicComponent,int)
COMPONENT_APPLY_BUILTIN(OpPeriodicComponent,unsigned)
COMPONENT_APPLY_BUILTIN(OpPeriodicComponent,short)
COMPONENT_APPLY_BUILTIN(OpPeriodicComponent,long)
COMPONENT_APPLY_BUILTIN(OpPeriodicComponent,float)
COMPONENT_APPLY_BUILTIN(OpPeriodicComponent,double)
COMPONENT_APPLY_BUILTIN(OpPeriodicComponent,std::complex<double>)


//////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// For unspecified centering, can't implement PeriodicFace::apply()
// correctly, and can't partial-specialize yet, so... don't have a prototype
// for unspecified centering, so user gets a compile error if he tries to
// invoke it for a centering not yet implemented. Implement external functions
// which are specializations for the various centerings
// {Cell,Vert,CartesianCentering}; these are called from the general
// PeriodicFace::apply() function body.
//----------------------------------------------------------------------------


//BENI: Do the whole operation part with += for Interpolation Boundary Conditions
//////////////////////////////////////////////////////////////////////

// Applicative templates for PeriodicFace:

// Standard, for applying to all components of elemental type:
// (N.B.: could just use OpAssign, but put this in for clarity and consistency
// with other appliciative templates in this module.)
template<class T>
struct OpInterpolation
{
};
template<class T>
inline void PETE_apply(const OpInterpolation<T>& /*e*/, T& a, const T& b) {a = a + b; }

// Special, for applying to single component of multicomponent elemental type:
template<class T>
struct OpInterpolationComponent
{
  OpInterpolationComponent(int c) : Component(c) {}
  int Component;
};

template<class T>
inline void PETE_apply(const OpInterpolationComponent<T>& e, T& a, const T& b)
{ a[e.Component] = a[e.Component]+b[e.Component]; }

// Following specializations are necessary because of the runtime branches in
// code which unfortunately force instantiation of OpPeriodicComponent
// instances for non-multicomponent types like {char,double,...}.
// Note: if user uses non-multicomponent (no operator[]) types of his own,
// he'll get a compile error. See comments regarding similar specializations
// for OpExtrapolateComponent for a more details.


COMPONENT_APPLY_BUILTIN(OpInterpolationComponent,char)
COMPONENT_APPLY_BUILTIN(OpInterpolationComponent,bool)
COMPONENT_APPLY_BUILTIN(OpInterpolationComponent,int)
COMPONENT_APPLY_BUILTIN(OpInterpolationComponent,unsigned)
COMPONENT_APPLY_BUILTIN(OpInterpolationComponent,short)
COMPONENT_APPLY_BUILTIN(OpInterpolationComponent,long)
COMPONENT_APPLY_BUILTIN(OpInterpolationComponent,float)
COMPONENT_APPLY_BUILTIN(OpInterpolationComponent,double)
COMPONENT_APPLY_BUILTIN(OpInterpolationComponent,std::complex<double>)
//////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// For unspecified centering, can't implement PeriodicFace::apply()
// correctly, and can't partial-specialize yet, so... don't have a prototype
// for unspecified centering, so user gets a compile error if he tries to
// invoke it for a centering not yet implemented. Implement external functions
// which are specializations for the various centerings
// {Cell,Vert,CartesianCentering}; these are called from the general
// PeriodicFace::apply() function body.
//----------------------------------------------------------------------------


template<class T, unsigned D, class M>
void PeriodicFaceBCApply(PeriodicFace<T,D,M,Cell>& pf,
			 Field<T,D,M,Cell>& A );
//BENI adds InterpolationFace ONLY Cell centered implementation!!!
template<class T, unsigned D, class M>
void InterpolationFaceBCApply(InterpolationFace<T,D,M,Cell>& pf,
			 Field<T,D,M,Cell>& A );

template<class T, unsigned D, class M>
void PeriodicFaceBCApply(PeriodicFace<T,D,M,Vert>& pf,
			 Field<T,D,M,Vert>& A );
template<class T, unsigned D, class M>
void PeriodicFaceBCApply(PeriodicFace<T,D,M,Edge>& pf,
			 Field<T,D,M,Edge>& A );
template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
void PeriodicFaceBCApply(PeriodicFace<T,D,M,
			 CartesianCentering<CE,D,NC> >& pf,
			 Field<T,D,M,CartesianCentering<CE,D,NC> >& A );

template<class T, unsigned D, class M, class C>
void PeriodicFace<T,D,M,C>::apply( Field<T,D,M,C>& A )
{

	//std::cout << "(2) PeriodicFace::apply" << std::endl;

  PeriodicFaceBCApply(*this, A);
}
//BENI adds InterpolationFace
template<class T, unsigned D, class M, class C>
void InterpolationFace<T,D,M,C>::apply( Field<T,D,M,C>& A )
{


  InterpolationFaceBCApply(*this, A);
}

//-----------------------------------------------------------------------------
// Specialization of PeriodicFace::apply() for Cell centering.
// Rather, indirectly-called specialized global function PeriodicFaceBCApply
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M>
void PeriodicFaceBCApply(PeriodicFace<T,D,M,Cell>& pf,
			 Field<T,D,M,Cell>& A )
{


	//std::cout << "(3) PeriodicFaceBCApply called" << std::endl;
  // NOTE: See the PeriodicFaceBCApply functions below for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );




  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = pf.getFace()/2;
  int offset;
  if ( pf.getFace() & 1 )
    {
      slab[d] = Index( domain[d].max() + 1, domain[d].max() + A.leftGuard(d) );
      offset = -domain[d].length();
    }
  else
    {
      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
      offset = domain[d].length();
    }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,Cell>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = src[d] + offset;

          // Loop over the ones that src touches.
          typename Field<T,D,M,Cell>::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
                  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  // BrickExpression<D,LFI,LFI,OpAssign >(lhs,rhs).apply();
		  if (pf.getComponent() == BCondBase<T,D,M,Cell>::allComponents) {
		    BrickExpression<D,LFI,LFI,OpPeriodic<T> >
		      (lhs,rhs,OpPeriodic<T>()).apply();
		  } else {
		    BrickExpression<D,LFI,LFI,OpPeriodicComponent<T> >
		      (lhs,rhs,OpPeriodicComponent<T>(pf.getComponent())).apply();
		  }
                }
            }
        }
    }
}

//BENI adds for InterpolationFace
//-----------------------------------------------------------------------------
// Specialization of InterpolationFace::apply() for Cell centering.
// Rather, indirectly-called specialized global function InerpolationFaceBCApply
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M>
void InterpolationFaceBCApply(InterpolationFace<T,D,M,Cell>& pf,
			 Field<T,D,M,Cell>& A )
{

  // NOTE: See the PeriodicFaceBCApply functions below for more
  // comprehensible comments --TJW

  // Find the slab that is the source (BENI: opposite to periodic BC).
  const NDIndex<D>& domain( A.getDomain() );

  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = pf.getFace()/2;
  int offset;
  if ( pf.getFace() & 1 )
    {
      slab[d] = Index( domain[d].max() + 1, domain[d].max() + A.leftGuard(d) );
      offset = -domain[d].length();
    }
  else
    {
      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
      offset = domain[d].length();
    }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,Cell>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          //BENI: The ghost values are the source to be accumulated to the boundaries
		  NDIndex<D> src = slab.intersect( fill_alloc );

          // BENI: destination is the boundary on the other side
          NDIndex<D> dest = src;
          dest[d] = dest[d] + offset;
		 // std::cout << "src = " << src << std::endl;
		 // std::cout << "dest = " << dest << std::endl;


          // Loop over the ones that src touches.
          typename Field<T,D,M,Cell>::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // BENI: If destination touches this LField...
              if ( dest.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
                  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  // BrickExpression<D,LFI,LFI,OpAssign >(lhs,rhs).apply();
		  if (pf.getComponent() == BCondBase<T,D,M,Cell>::allComponents) {
			  //std::cout << "TRY to apply OPInterpol" << std::endl;
		    BrickExpression<D,LFI,LFI,OpInterpolation<T> >
		      (lhs,rhs,OpInterpolation<T>()).apply();
		  } else {
		    BrickExpression<D,LFI,LFI,OpInterpolationComponent<T> >
		      (lhs,rhs,OpInterpolationComponent<T>(pf.getComponent())).apply();
		  }
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Specialization of PeriodicFace::apply() for Vert centering.
// Rather, indirectly-called specialized global function PeriodicFaceBCApply
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M>
void PeriodicFaceBCApply(PeriodicFace<T,D,M,Vert>& pf,
			 Field<T,D,M,Vert>& A )
{



  // NOTE: See the ExtrapolateFaceBCApply functions below for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = pf.getFace()/2;
  int offset;
  if ( pf.getFace() & 1 )
    {
      // TJW: this used to say "leftGuard(d)", which I think was wrong:
      slab[d] = Index(domain[d].max(), domain[d].max() + A.rightGuard(d));
      // N.B.: the extra +1 here is what distinguishes this Vert-centered
      // implementation from the Cell-centered one:
      offset = -domain[d].length() + 1;
    }
  else
    {
      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
      // N.B.: the extra -1 here is what distinguishes this Vert-centered
      // implementation from the Cell-centered one:
      offset = domain[d].length() - 1;
    }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,Vert>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = src[d] + offset;

          // Loop over the ones that src touches.
          typename Field<T,D,M,Vert>::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
                  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  // BrickExpression<D,LFI,LFI,OpAssign >(lhs,rhs).apply();
		  if (pf.getComponent() == BCondBase<T,D,M,Vert>::allComponents) {
		    BrickExpression<D,LFI,LFI,OpPeriodic<T> >
		      (lhs,rhs,OpPeriodic<T>()).apply();
		  } else {
		    BrickExpression<D,LFI,LFI,OpPeriodicComponent<T> >
		      (lhs,rhs,OpPeriodicComponent<T>(pf.getComponent())).apply();
		  }
                }
            }
        }
    }
}


//-----------------------------------------------------------------------------
// Specialization of PeriodicFace::apply() for Edge centering.
// Rather, indirectly-called specialized global function PeriodicFaceBCApply
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M>
void PeriodicFaceBCApply(PeriodicFace<T,D,M,Edge>& pf,
			 Field<T,D,M,Edge>& A )
{
  // NOTE: See the ExtrapolateFaceBCApply functions below for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = pf.getFace()/2;
  int offset;
  if ( pf.getFace() & 1 )
    {
      // TJW: this used to say "leftGuard(d)", which I think was wrong:
      slab[d] = Index(domain[d].max(), domain[d].max() + A.rightGuard(d));
      // N.B.: the extra +1 here is what distinguishes this Edge-centered
      // implementation from the Cell-centered one:
      offset = -domain[d].length() + 1;
    }
  else
    {
      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
      // N.B.: the extra -1 here is what distinguishes this Edge-centered
      // implementation from the Cell-centered one:
      offset = domain[d].length() - 1;
    }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,Edge>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = src[d] + offset;

          // Loop over the ones that src touches.
          typename Field<T,D,M,Edge>::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
                  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  // BrickExpression<D,LFI,LFI,OpAssign >(lhs,rhs).apply();
		  if (pf.getComponent() == BCondBase<T,D,M,Edge>::allComponents) {
		    BrickExpression<D,LFI,LFI,OpPeriodic<T> >
		      (lhs,rhs,OpPeriodic<T>()).apply();
		  } else {
		    BrickExpression<D,LFI,LFI,OpPeriodicComponent<T> >
		      (lhs,rhs,OpPeriodicComponent<T>(pf.getComponent())).apply();
		  }
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Specialization of PeriodicFace::apply() for CartesianCentering centering.
// Rather, indirectly-called specialized global function PeriodicFaceBCApply
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
void PeriodicFaceBCApply(PeriodicFace<T,D,M,
			 CartesianCentering<CE,D,NC> >& pf,
			 Field<T,D,M,CartesianCentering<CE,D,NC> >& A )
{



  // NOTE: See the ExtrapolateFaceBCApply functions below for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = pf.getFace()/2;
  int offset;
  if ( pf.getFace() & 1 )
    {
      // Do the right thing for CELL or VERT centering for this component (or
      // all components, if the PeriodicFace object so specifies):
      if (pf.getComponent() == BCondBase<T,D,M,CartesianCentering<CE,D,NC> >::
	  allComponents) {
	// Make sure all components are really centered the same, as assumed:
	CenteringEnum centering0 = CE[0 + d*NC]; // 1st component along dir d
	for (unsigned int c=1; c<NC; c++) { // Compare other components with 1st
	  if (CE[c + d*NC] != centering0)
	    ERRORMSG("PeriodicFaceBCApply: BCond thinks all components have"
		     << " same centering along direction " << d
		     << ", but it isn't so." << endl);
	}
	// Now do the right thing for CELL or VERT centering of all components:
	if (centering0 == CELL) {
	  offset = -domain[d].length();     // CELL case
	} else {
	  // TJW: this used to say "leftGuard(d)", which I think was wrong:
	  slab[d] =
	    Index( domain[d].max(), domain[d].max() + A.rightGuard(d));
	  offset = -domain[d].length()+1; // VERT case
	}
      } else { // The BC applies only to one component, not all:
	// Do the right thing for CELL or VERT centering of the component:
	if (CE[pf.getComponent() + d*NC] == CELL) {
	  offset = -domain[d].length();     // CELL case
	} else {
	  slab[d] =
	    Index( domain[d].max(), domain[d].max() + A.rightGuard(d));
	  offset = -domain[d].length()+1; // VERT case
	}
      }
    }
  else
    {
      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
      // Do the right thing for CELL or VERT centering for this component (or
      // all components, if the PeriodicFace object so specifies):
      if (pf.getComponent() == BCondBase<T,D,M,CartesianCentering<CE,D,NC> >::
	  allComponents) {
	// Make sure all components are really centered the same, as assumed:
	CenteringEnum centering0 = CE[0 + d*NC]; // 1st component along dir d
	for (unsigned int c=1; c<NC; c++) { // Compare other components with 1st
	  if (CE[c + d*NC] != centering0)
	    ERRORMSG("PeriodicFaceBCApply: BCond thinks all components have"
		     << " same centering along direction " << d
		     << ", but it isn't so." << endl);
	}
	// Now do the right thing for CELL or VERT centering of all components:
	if (centering0 == CELL) {
	  offset = -domain[d].length();     // CELL case
	} else {
	  offset = -domain[d].length() + 1; // VERT case
	}
      } else { // The BC applies only to one component, not all:
	// Do the right thing for CELL or VERT centering of the component:
	if (CE[pf.getComponent() + d*NC] == CELL) {
	  offset = domain[d].length();     // CELL case
	} else {
	  offset = domain[d].length() - 1; // VERT case
	}
      }
    }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,CartesianCentering<CE,D,NC> >::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = src[d] + offset;

          // Loop over the ones that src touches.
          typename Field<T,D,M,CartesianCentering<CE,D,NC> >::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
                  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  // BrickExpression<D,LFI,LFI,OpAssign >(lhs,rhs).apply();
		  if (pf.getComponent() == BCondBase<T,D,M,
		      CartesianCentering<CE,D,NC> >::allComponents) {
		    BrickExpression<D,LFI,LFI,OpPeriodic<T> >
		      (lhs,rhs,OpPeriodic<T>()).apply();
		  } else {
		    BrickExpression<D,LFI,LFI,OpPeriodicComponent<T> >
		      (lhs,rhs,OpPeriodicComponent<T>(pf.getComponent())).apply();
		  }
                }
            }
        }
    }
}


//-----------------------------------------------------------------------------
// Specialization of CalcParallelPeriodicDomain for various centerings.
// This is the centering-specific code for ParallelPeriodicFace::apply().
//-----------------------------------------------------------------------------

#ifdef PRINT_DEBUG
// For distance.
#  include <iterator.h>
#endif

template <class T, unsigned D, class M>
inline void
CalcParallelPeriodicDomain(const Field<T,D,M,Cell> &A,
			   const ParallelPeriodicFace<T,D,M,Cell>& pf,
			   NDIndex<D> &dest_slab,
			   int &offset)
{
  // Direction Dim has faces 2*Dim and 2*Dim + 1, so the following
  // expression converts the face index to the direction index.

  unsigned d = pf.getFace()/2;

  const NDIndex<D>& domain(A.getDomain());

  if (pf.getFace() & 1) // Odd ("top" or "right") face
    {
      // The cells that we need to fill start one beyond the last
      // physical cell at the "top" and continue to the last guard
      // cell. Change "dest_slab" to restrict direction "d" to this
      // subdomain.

      dest_slab[d] =
	Index(domain[d].max() + 1, domain[d].max() + A.leftGuard(d));

      // The offset to the cells that we are going to read; i.e. the
      // read domain will be "dest_slab + offset". This is the number of
      // physical cells in that direction.

      offset = -domain[d].length();
    }
  else // Even ("bottom" or "left") face
    {
      // The cells that we need to fill start with the first guard
      // cell on the bottom and continue up through the cell before
      // the first physical cell.

      dest_slab[d] =
	Index(domain[d].min() - A.leftGuard(d), domain[d].min()-1);

      // See above.

      offset = domain[d].length();
    }
}

// Note: this does the same thing that PeriodicFace::apply() does, but
// I don't think that this is correct.

template <class T, unsigned D, class M>
inline void
CalcParallelPeriodicDomain(const Field<T,D,M,Vert> &A,
			   const ParallelPeriodicFace<T,D,M,Vert>& pf,
			   NDIndex<D> &dest_slab,
			   int &offset)
{
  // Direction Dim has faces 2*Dim and 2*Dim + 1, so the following
  // expression converts the face index to the direction index.

  const NDIndex<D>& domain(A.getDomain());

  unsigned d = pf.getFace()/2;

  if (pf.getFace() & 1) // Odd ("top" or "right") face
    {
      // A vert-centered periodic field duplicates the boundary
      // point. As a result, the right boundary point is filled from
      // the left boundary point. Thus, the points that we need to fill
      // include the last physical point (domain[d].max()) and the
      // guard points.

      dest_slab[d] =
	Index(domain[d].max(), domain[d].max() + A.rightGuard(d));

      // The offset to the points that we are going to read; i.e. the
      // read domain will be "dest_slab + offset". This is the number of
      // physical points in that direction.

      offset = -domain[d].length() + 1;
    }
  else // Even ("bottom" or "left") face
    {
      // The points that we need to fill start with the first guard
      // cell on the bottom and continue up through the cell before
      // the first physical cell.

      dest_slab[d] =
	Index(domain[d].min() - A.leftGuard(d), domain[d].min()-1);

      // See above.

      offset = domain[d].length() - 1;
    }
}

// See comments above - vert centering wrong, I think.
// TODO ckr: compare this with the general case below
template <class T, unsigned D, class M>
inline void
CalcParallelPeriodicDomain(const Field<T,D,M,Edge> &A,
			   const ParallelPeriodicFace<T,D,M,Edge>& pf,
			   NDIndex<D> &dest_slab,
			   int &offset)
{
  // Direction Dim has faces 2*Dim and 2*Dim + 1, so the following
  // expression converts the face index to the direction index.

  const NDIndex<D>& domain(A.getDomain());

  unsigned d = pf.getFace()/2;

  if (pf.getFace() & 1) // Odd ("top" or "right") face
    {
      // A vert-centered periodic field duplicates the boundary
      // point. As a result, the right boundary point is filled from
      // the left boundary point. Thus, the points that we need to fill
      // include the last physical point (domain[d].max()) and the
      // guard points.

      dest_slab[d] =
	Index(domain[d].max(), domain[d].max() + A.rightGuard(d));

      // The offset to the points that we are going to read; i.e. the
      // read domain will be "dest_slab + offset". This is the number of
      // physical points in that direction.

      offset = -domain[d].length() + 1;
    }
  else // Even ("bottom" or "left") face
    {
      // The points that we need to fill start with the first guard
      // cell on the bottom and continue up through the cell before
      // the first physical cell.

      dest_slab[d] =
	Index(domain[d].min() - A.leftGuard(d), domain[d].min()-1);

      // See above.

      offset = domain[d].length() - 1;
    }
}

// See comments above - vert centering wrong, I think.

template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
inline void
CalcParallelPeriodicDomain(const Field<T,D,M,CartesianCentering<CE,D,NC> >& A,
			   const ParallelPeriodicFace<T,D,M,
			           CartesianCentering<CE,D,NC> >& pf,
			   NDIndex<D> &dest_slab,
			   int &offset)
{
  typedef BCondBase<T,D,M,CartesianCentering<CE,D,NC> > BCBase_t;

  // Direction Dim has faces 2*Dim and 2*Dim + 1, so the following
  // expression converts the face index to the direction index.

  const NDIndex<D>& domain(A.getDomain());

  unsigned d = pf.getFace()/2;

  if (pf.getFace() & 1) // Odd ("top" or "right") face
    {
      // For this specialization we need to do the right thing for
      // CELL or VERT centering for the appropriate components of the
      // field. The cells that we need to fill, and the offset to the
      // source cells, depend on the centering.  See below and the
      // comments in the vert and cell versions above.

      if (pf.getComponent() == BCBase_t::allComponents)
	{
	  // Make sure all components are really centered the same, as
	  // assumed:

	  CenteringEnum centering0 = CE[0 + d*NC]; // 1st component
	                                           // along dir d
	  for (unsigned int c = 1; c < NC; c++)
	    {
	      // Compare other components with 1st
	      if (CE[c + d*NC] != centering0)
		ERRORMSG("ParallelPeriodicFaceBCApply:"
			 << "BCond thinks all components have"
			 << " same centering along direction " << d
			 << ", but it isn't so." << endl);
	    }

	  // Now do the right thing for CELL or VERT centering of all
	  // components:

	  if (centering0 == CELL) {
	    offset = -domain[d].length();     // CELL case
	  } else {
	    dest_slab[d] =
	      Index(domain[d].max(), domain[d].max() + A.leftGuard(d));
	    offset = -domain[d].length() + 1; // VERT case
	  }

	}
      else
	{
	  // The BC applies only to one component, not all: Do the
	  // right thing for CELL or VERT centering of the component:

	  if (CE[pf.getComponent() + d*NC] == CELL)
	    {
	      offset = -domain[d].length();     // CELL case
	    }
	  else
	    {
	      dest_slab[d] =
		Index(domain[d].max(), domain[d].max() + A.leftGuard(d));
	      offset = -domain[d].length() + 1; // VERT case
	    }
	}
    }
  else // Even ("bottom" or "left") face
    {
      // The cells that we need to fill start with the first guard
      // cell on the bottom and continue up through the cell before
      // the first physical cell.

      dest_slab[d] =
	Index(domain[d].min() - A.leftGuard(d), domain[d].min()-1);

      // See above.

      if (pf.getComponent() == BCBase_t::allComponents)
	{
	  // Make sure all components are really centered the same, as
	  // assumed:

	  CenteringEnum centering0 = CE[0 + d*NC]; // 1st component
	                                           // along dir d
	  for (unsigned int c = 1; c < NC; c++)
	    { // Compare other components with 1st
	      if (CE[c + d*NC] != centering0)
		ERRORMSG("ParallelPeriodicFaceBCApply:"
			 << "BCond thinks all components have"
			 << " same centering along direction " << d
			 << ", but it isn't so." << endl);
	    }

	  // Now do the right thing for CELL or VERT centering of all
	  // components:

	  if (centering0 == CELL) {
	    offset = -domain[d].length();     // CELL case
	  } else {
	    offset = -domain[d].length() + 1; // VERT case
	  }

	}
      else
	{
	  // The BC applies only to one component, not all: Do the
	  // right thing for CELL or VERT centering of the component:

	  if (CE[pf.getComponent() + d*NC] == CELL)
	    {
	      offset = domain[d].length();     // CELL case
	    }
	  else
	    {
	      offset = domain[d].length() - 1; // VERT case
	    }
	}
    }
}

//-----------------------------------------------------------------------------
// ParallelPeriodicFace::apply()
// Apply the periodic boundary condition. This version can handle
// fields that are parallel in the periodic direction. Unlike the
// other BCond types, the Lion's share of the code is in this single
// apply() method. The only centering-specific calculation is that of
// the destination domain and the offset, and that is separated out
// into the CalcParallelPeriodicDomain functions defined above.
//-----------------------------------------------------------------------------
//#define PRINT_DEBUG
template<class T, unsigned D, class M, class C>
void ParallelPeriodicFace<T,D,M,C>::apply( Field<T,D,M,C>& A )
{

#ifdef PRINT_DEBUG
  Inform msg("PPeriodicBC", INFORM_ALL_NODES);
#endif


  // Useful typedefs:

  typedef T                   Element_t;
  typedef NDIndex<D>          Domain_t;
  typedef LField<T,D>         LField_t;
  typedef typename LField_t::iterator  LFI_t;
  typedef Field<T,D,M,C>      Field_t;
  typedef FieldLayout<D>      Layout_t;

  //===========================================================================
  //
  // Unlike most boundary conditions, periodic BCs are (in general)
  // non-local. Indeed, they really are identical to the guard-cell
  // seams between LFields internal to the Field. In this case the
  // LFields just have a periodic geometry, but the FieldLayout
  // doesn't express this, so we duplicate the code, which is quite
  // similar to fillGuardCellsr, but somewhat more complicated, here.
  // The complications arise from three sources:
  //
  //  - The source and destination domains are offset, not overlapping.
  //  - Only a subset of all LFields are, in general, involved.
  //  - The corners must be handled correctly.
  //
  // Here's the plan:
  //
  //  0. Calculate source and destination domains.
  //  1. Build send and receive lists, and send messages.
  //  2. Evaluate local pieces directly.
  //  3. Receive messages and evaluate remaining pieces.
  //
  //===========================================================================
/*
#ifdef PRINT_DEBUG
  msg << "Starting BC Calculation for face "
      << getFace() << "." << endl;
#endif
*/
  //===========================================================================
  //  0. Calculate destination domain and the offset.
  //===========================================================================

  // Find the slab that is the destination. First get the whole
  // domain, including guard cells, and then restrict it to the area
  // that this BC will fill.

  const NDIndex<D>& domain(A.getDomain());

  NDIndex<D> dest_slab = AddGuardCells(domain,A.getGuardCellSizes());

  // Direction Dim has faces 2*Dim and 2*Dim + 1, so the following
  // expression converts the face index to the direction index.

  unsigned d = this->getFace()/2;

  int offset;

  CalcParallelPeriodicDomain(A,*this,dest_slab,offset);

  Domain_t src_slab = dest_slab;
  src_slab[d] = src_slab[d] + offset;

#ifdef PRINT_DEBUG
  msg << "dest_slab = " << dest_slab << endl;
  msg << "src_slab  = " << src_slab  << endl;
  //  stop_here();
#endif


  //===========================================================================
  //  1. Build send and receive lists and send messages
  //===========================================================================

  // Declare these at this scope so that we don't have to duplicate
  // the local code. (fillguardcells puts these in the scope of the
  // "if (nprocs > 1) { ... }" section, but has to duplicate the
  // code for the local fills as a result. The cost of this is a bit
  // of stackspace, and the cost of allocating an empty map.)

  // Container for holding Domain -> LField mapping
  // so that we can sort out which messages go where.

  typedef std::multimap<Domain_t,LField_t*, std::less<Domain_t> > ReceiveMap_t;

  // (Time this since it allocates an empty map.)



  ReceiveMap_t receive_map;



  // Number of nodes that will send us messages.

  int receive_count = 0;
  int send_count = 0;

  // Communications tag

  int bc_comm_tag;


  // Next fill the dest_list and src_list, lists of the LFields that
  // touch the destination and source domains, respectively.

  // (Do we need this for local-only code???)

  // (Also, if a domain ends up in both lists, it will only be
  // involved in local communication. We should structure this code to
  // take advantage of this, otherwise all existing parallel code is
  // going to incur additional overhead that really is unnecessary.)
  // (In other words, we should be able to do the general case, but
  // this capability shouldn't slow down the typical cases too much.)

  typedef std::vector<LField_t*> DestList_t;
  typedef std::vector<LField_t*> SrcList_t;
  typedef typename DestList_t::iterator DestListIterator_t;
  typedef typename SrcList_t::iterator SrcListIterator_t;

  DestList_t dest_list;
  SrcList_t src_list;

  dest_list.reserve(1);
  src_list.reserve(1);

  typename Field_t::iterator_if lf_i;

#ifdef PRINT_DEBUG
  msg << "Starting dest & src domain calculation." << endl;
#endif

  for (lf_i = A.begin_if(); lf_i != A.end_if(); ++lf_i)
    {
      LField_t &lf = *lf_i->second;

      // We fill if our allocated domain touches the
      // destination slab.

      const Domain_t &lf_allocated = lf.getAllocated();

#ifdef PRINT_DEBUG
      msg << "  Processing subdomain : " << lf_allocated << endl;
      //      stop_here();
#endif

      if (lf_allocated.touches(dest_slab))
	dest_list.push_back(&lf);

      // We only provide data if our owned cells touch
      // the source slab (although we actually send the
      // allocated data).

      const Domain_t &lf_owned = lf.getOwned();

      if (lf_owned.touches(src_slab))
	src_list.push_back(&lf);
    }

#ifdef PRINT_DEBUG
  msg << "  dest_list has " << dest_list.size() << " elements." << endl;
  msg << "  src_list has " << src_list.size() << " elements." << endl;
#endif

  DestListIterator_t dest_begin = dest_list.begin();
  DestListIterator_t dest_end   = dest_list.end();
  SrcListIterator_t src_begin  = src_list.begin();
  SrcListIterator_t src_end    = src_list.end();

  // Aliases to some of Field A's properties...

  const Layout_t &layout      = A.getLayout();
  const GuardCellSizes<D> &gc = A.getGuardCellSizes();

  int nprocs = Ippl::getNodes();

  if (nprocs > 1) // Skip send/receive code if we're single-processor.
    {


#ifdef PRINT_DEBUG
      msg << "Starting receive calculation." << endl;
      //      stop_here();
#endif

      //---------------------------------------------------
      // Receive calculation
      //---------------------------------------------------

      // Mask indicating the nodes will send us messages.

      std::vector<bool> receive_mask(nprocs,false);

      DestListIterator_t dest_i;

      for (dest_i = dest_begin; dest_i != dest_end; ++dest_i)
        {
          // Cache some information about this local array.

          LField_t &dest_lf = **dest_i;

          const Domain_t &dest_lf_alloc = dest_lf.getAllocated();

	  // Calculate the destination domain in this LField, and the
	  // source domain (which may be spread across multipple
	  // processors) from whence that domain will be filled:

	  const Domain_t dest_domain = dest_lf_alloc.intersect(dest_slab);

	  Domain_t src_domain = dest_domain;
	  src_domain[d] = src_domain[d] + offset;

          // Find the remote LFields that contain src_domain. Note
          // that we have to fill from the full allocated domains in
          // order to properly fill the corners of the boundary cells,
          // BUT we only need to intersect with the physical domain.
          // Intersecting the allocated domain would result in
          // unnecessary messages. (In fact, only the corners *need* to
          // send the allocated domains, but for regular decompositions,
          // sending the allocated domains will result in fewer
          // messages [albeit larger ones] than sending only from
          // physical cells.)

          typename Layout_t::touch_range_dv
            src_range(layout.touch_range_rdv(src_domain));

	  // src_range is a begin/end pair into a list of remote
	  // domain/vnode pairs whose physical domains touch
	  // src_domain. Iterate through this list and set up the
	  // receive map and the receive mask.

          typename Layout_t::touch_iterator_dv rv_i;

          for (rv_i = src_range.first; rv_i != src_range.second; ++rv_i)
            {
              // Intersect src_domain with the allocated cells for the
	      // remote LField (rv_alloc). This will give us the strip
	      // that will be sent. Translate this domain back to the
	      // domain of the receiving LField.

	      const Domain_t rv_alloc = AddGuardCells(rv_i->first,gc);

              Domain_t hit = src_domain.intersect(rv_alloc);
	      hit[d] = hit[d] - offset;

	      // Save this domain and the LField pointer

              typedef typename ReceiveMap_t::value_type value_type;

              receive_map.insert(value_type(hit,&dest_lf));

#ifdef PRINT_DEBUG
	      msg << "  Need remote data for domain: " << hit << endl;
#endif

              // Note who will be sending this data

              int rnode = rv_i->second->getNode();

              receive_mask[rnode] = true;

            } // rv_i
	} // dest_i

      receive_count = 0;

      for (int iproc = 0; iproc < nprocs; ++iproc)
	if (receive_mask[iproc]) ++receive_count;


#ifdef PRINT_DEBUG
      msg << "  Expecting to see " << receive_count << " messages." << endl;
      msg << "Done with receive calculation." << endl;
      //      stop_here();
#endif






#ifdef PRINT_DEBUG
      msg << "Starting send calculation" << endl;
#endif

      //---------------------------------------------------
      // Send calculation
      //---------------------------------------------------

      // Array of messages to be sent.

      std::vector<Message *> messages(nprocs);
      for (int miter=0; miter < nprocs; messages[miter++] = 0);

      // Debugging info.

#ifdef PRINT_DEBUG
      // KCC 3.2d has trouble with this. 3.3 doesn't, but
      // some are still using 3.2.
      //      vector<int> ndomains(nprocs,0);
      std::vector<int> ndomains(nprocs);
      for(int i = 0; i < nprocs; ++i) ndomains[i] = 0;
#endif

      SrcListIterator_t src_i;

      for (src_i = src_begin; src_i != src_end; ++src_i)
        {
          // Cache some information about this local array.

          LField_t &src_lf = **src_i;

	  // We need to send the allocated data to properly fill the
	  // corners when using periodic BCs in multipple dimensions.
	  // However, we don't want to send to nodes that only would
	  // receive data from our guard cells. Thus we do the
	  // intersection test with our owned data.

          const Domain_t &src_lf_owned = src_lf.getOwned();
	  const Domain_t &src_lf_alloc = src_lf.getAllocated();

	  // Calculate the owned and allocated parts of the source
	  // domain in this LField, and corresponding destination
	  // domains.

	  const Domain_t src_owned = src_lf_owned.intersect(src_slab);
	  const Domain_t src_alloc = src_lf_alloc.intersect(src_slab);

	  Domain_t dest_owned = src_owned;
	  dest_owned[d] = dest_owned[d] - offset;

	  Domain_t dest_alloc = src_alloc;
	  dest_alloc[d] = dest_alloc[d] - offset;

#ifdef PRINT_DEBUG
	  msg << "  Considering LField with the domains:" << endl;
	  msg << "     owned = " << src_lf_owned << endl;
	  msg << "     alloc = " << src_lf_alloc << endl;
	  msg << "  The intersections with src_slab are:" << endl;
	  msg << "     owned = " << src_owned << endl;
	  msg << "     alloc = " << src_alloc << endl;
#endif

          // Find the remote LFields whose allocated cells (note the
	  // additional "gc" arg) are touched by dest_owned.

          typename Layout_t::touch_range_dv
            dest_range(layout.touch_range_rdv(dest_owned,gc));

          typename Layout_t::touch_iterator_dv rv_i;
/*
#ifdef PRINT_DEBUG
	  msg << "  Touch calculation found "
	      << distance(dest_range.first, dest_range.second)
	      << " elements." << endl;
#endif
*/

          for (rv_i = dest_range.first; rv_i != dest_range.second; ++rv_i)
            {
              // Find the intersection of the returned domain with the
	      // allocated version of the translated source domain.
	      // Translate this intersection back to the source side.

              Domain_t hit = dest_alloc.intersect(rv_i->first);
	      hit[d] = hit[d] + offset;

              // Find out who owns this remote domain.

              int rnode = rv_i->second->getNode();

#ifdef PRINT_DEBUG
              msg << "  Overlap domain = " << rv_i->first << endl;
              msg << "  Inters. domain = " << hit;
              msg << "  --> node " << rnode << endl;
#endif

              // Create an LField iterator for this intersection region,
              // and try to compress it. (Copied from fillGuardCells -
	      // not quite sure how this works yet. JAC)

              // storage for LField compression

              Element_t compressed_value;
              LFI_t msgval = src_lf.begin(hit, compressed_value);
              msgval.TryCompress();

              // Put intersection domain and field data into message

              if (!messages[rnode])
		{
		  messages[rnode] = new Message;
		  PAssert(messages[rnode]);
		}

              messages[rnode]->put(hit);    // puts Dim items in Message
              messages[rnode]->put(msgval); // puts 3 items in Message

#ifdef PRINT_DEBUG
              ndomains[rnode]++;
#endif
            }  // rv_i
	} // src_i

      // Get message tag.

      bc_comm_tag =
	Ippl::Comm->next_tag(BC_PARALLEL_PERIODIC_TAG,BC_TAG_CYCLE);



      // Send the messages.

      for (int iproc = 0; iproc < nprocs; ++iproc)
	{
	  if (messages[iproc])
	    {

#ifdef PRINT_DEBUG
	      msg << "  ParallelPeriodicBCApply: Sending message to node "
		  << iproc << endl
		  << "    number of domains  = " << ndomains[iproc] << endl
		  << "    number of MsgItems = "
		  << messages[iproc]->size() << endl;
#endif

	      Ippl::Comm->send(messages[iproc], iproc, bc_comm_tag);
	      ++send_count;

	    }

	}

#ifdef PRINT_DEBUG
      msg << "  Sent " << send_count << " messages" << endl;
      msg << "Done with send." << endl;
#endif





    } // if (nprocs > 1)




  //===========================================================================
  //  2. Evaluate local pieces directly.
  //===========================================================================

#ifdef PRINT_DEBUG
  msg << "Starting local calculation." << endl;
#endif

  DestListIterator_t dest_i;

  for (dest_i = dest_begin; dest_i != dest_end; ++dest_i)
    {
      // Cache some information about this local array.

      LField_t &dest_lf = **dest_i;

      const Domain_t &dest_lf_alloc = dest_lf.getAllocated();

      const Domain_t dest_domain = dest_lf_alloc.intersect(dest_slab);

      Domain_t src_domain = dest_domain;
      src_domain[d] = src_domain[d] + offset;

      SrcListIterator_t src_i;

      for (src_i = src_begin; src_i != src_end; ++src_i)
        {
          // Cache some information about this local array.

          LField_t &src_lf = **src_i;

	  // Unlike fillGuardCells, we need to send the allocated
	  // data.  This is necessary to properly fill the corners
	  // when using periodic BCs in multipple dimensions.

          const Domain_t &src_lf_owned = src_lf.getOwned();
          const Domain_t &src_lf_alloc = src_lf.getAllocated();

	  // Only fill from LFields whose physical domain touches
	  // the translated destination domain.

	  if (src_domain.touches(src_lf_owned))
	    {
	      // Worry about compression. Should do this right
	      // (considering the four different combinations), but
	      // for now just do what the serial version does:

	      dest_lf.Uncompress();

	      // Calculate the actual source and destination domains.

	      Domain_t real_src_domain =
		src_domain.intersect(src_lf_alloc);

	      Domain_t real_dest_domain = real_src_domain;
	      real_dest_domain[d] = real_dest_domain[d] - offset;

#ifdef PRINT_DEBUG
	      msg << "  Copying local data . . ." << endl;
	      msg << "    source domain = " << real_src_domain << endl;
	      msg << "    dest domain   = " << real_dest_domain << endl;
#endif

	      // Build the iterators for the copy

	      LFI_t lhs = dest_lf.begin(real_dest_domain);
	      LFI_t rhs = src_lf.begin(real_src_domain);

	      // And do the assignment:

	      if (this->getComponent() == BCondBase<T,D,M,C>::allComponents)
		{
		  BrickExpression<D,LFI_t,LFI_t,OpPeriodic<T> >
		    (lhs,rhs,OpPeriodic<T>()).apply();
		}
	      else
		{
		  BrickExpression<D,LFI_t,LFI_t,OpPeriodicComponent<T> >
		    (lhs,rhs,OpPeriodicComponent<T>(this->getComponent())).apply();
		}
            }
        }
    }

#ifdef PRINT_DEBUG
  msg << "Done with local calculation." << endl;
#endif




  //===========================================================================
  //  3. Receive messages and evaluate remaining pieces.
  //===========================================================================

  if (nprocs > 1)
    {



#ifdef PRINT_DEBUG
      msg << "Starting receive..." << endl;
      //      stop_here();
#endif

      while (receive_count > 0)
	{

	  // Receive the next message.

	  int any_node = COMM_ANY_NODE;



	  Message* message =
	    Ippl::Comm->receive_block(any_node,bc_comm_tag);
	  PAssert(message);

	  --receive_count;



	  // Determine the number of domains being sent

	  int ndomains = message->size() / (D + 3);

#ifdef PRINT_DEBUG
	  msg << "  Message received from node "
	      << any_node << "," << endl
	      << "  number of domains = " << ndomains << endl;
#endif

	  for (int idomain=0; idomain < ndomains; ++idomain)
	    {
	      // Extract the intersection domain from the message
	      // and translate it to the destination side.

	      Domain_t intersection;
	      intersection.getMessage(*message);
	      intersection[d] = intersection[d] - offset;

	      // Extract the rhs iterator from it.

	      Element_t compressed_value;
	      LFI_t rhs(compressed_value);
	      rhs.getMessage(*message);

#ifdef PRINT_DEBUG
	      msg << "  Received remote overlap region = "
		  << intersection << endl;
#endif

	      // Find the LField it is destined for.

	      typename ReceiveMap_t::iterator hit =
		receive_map.find(intersection);

	      PAssert(hit != receive_map.end());

	      // Build the lhs brick iterator.

	      // Should have been
	      // LField<T,D> &lf = *hit->second;
	      // but SGI's 7.2  multimap doesn't have op->().

	      LField<T,D> &lf = *(*hit).second;

	      // Check and see if we really have to do this.

#ifdef PRINT_DEBUG
	      msg << "   LHS compressed ? " << lf.IsCompressed();
	      msg << ", LHS value = " << *lf.begin() << endl;
	      msg << "   RHS compressed ? " << rhs.IsCompressed();
	      msg << ", RHS value = " << *rhs << endl;
	      msg << "   *rhs == *lf.begin() ? "
		  << (*rhs == *lf.begin()) << endl;
#endif
	      if (!(rhs.IsCompressed() && lf.IsCompressed() &&
		    (*rhs == *lf.begin())))
		{
		  // Yep. gotta do it.

		  lf.Uncompress();
		  LFI_t lhs = lf.begin(intersection);

		  // Do the assignment.

#ifdef PRINT_DEBUG
		  msg << "   Doing copy." << endl;
#endif

		  BrickExpression<D,LFI_t,LFI_t,OpAssign>(lhs,rhs).apply();
		}

	      // Take that entry out of the receive list.

	      receive_map.erase(hit);
	    }

	  delete message;
	}


#ifdef PRINT_DEBUG
      msg << "Done with receive." << endl;
#endif

    }
}


////////////////////////////////////////
// BENI adds CalcParallelInterpolationDomain
/////////////////////////////////////////////
template <class T, unsigned D, class M>
inline void
CalcParallelInterpolationDomain(const Field<T,D,M,Cell> &A,
			   const ParallelInterpolationFace<T,D,M,Cell>& pf,
			   NDIndex<D> &src_slab,
			   int &offset)
{
  // Direction Dim has faces 2*Dim and 2*Dim + 1, so the following
  // expression converts the face index to the direction index.

  unsigned d = pf.getFace()/2;

  const NDIndex<D>& domain(A.getDomain());

  if (pf.getFace() & 1) // Odd ("top" or "right") face
    {
      // The cells that we need to fill start one beyond the last
      // physical cell at the "top" and continue to the last guard
      // cell. Change "dest_slab" to restrict direction "d" to this
      // subdomain.

      src_slab[d] =
	Index(domain[d].max() + 1, domain[d].max() + A.leftGuard(d));

      // The offset to the cells that we are going to read; i.e. the
      // read domain will be "dest_slab + offset". This is the number of
      // physical cells in that direction.

      offset = -domain[d].length();
    }
  else // Even ("bottom" or "left") face
    {
      // The cells that we need to fill start with the first guard
      // cell on the bottom and continue up through the cell before
      // the first physical cell.

      src_slab[d] =
	Index(domain[d].min() - A.leftGuard(d), domain[d].min()-1);

      // See above.

      offset = domain[d].length();
    }
}


//////////////////////////////////////////////////////////////////////
//BENI adds parallelInterpo;lationBC apply
//////////////////////////////////////////////////////////////////////
template<class T, unsigned D, class M, class C>
void ParallelInterpolationFace<T,D,M,C>::apply( Field<T,D,M,C>& A )
{

#ifdef PRINT_DEBUG
  Inform msg("PInterpolationBC", INFORM_ALL_NODES);
#endif


  // Useful typedefs:

  typedef T                   Element_t;
  typedef NDIndex<D>          Domain_t;
  typedef LField<T,D>         LField_t;
  typedef typename LField_t::iterator  LFI_t;
  typedef Field<T,D,M,C>      Field_t;
  typedef FieldLayout<D>      Layout_t;

  //===========================================================================
  //
  // Unlike most boundary conditions, periodic BCs are (in general)
  // non-local. Indeed, they really are identical to the guard-cell
  // seams between LFields internal to the Field. In this case the
  // LFields just have a periodic geometry, but the FieldLayout
  // doesn't express this, so we duplicate the code, which is quite
  // similar to fillGuardCellsr, but somewhat more complicated, here.
  // The complications arise from three sources:
  //
  //  - The source and destination domains are offset, not overlapping.
  //  - Only a subset of all LFields are, in general, involved.
  //  - The corners must be handled correctly.
  //
  // Here's the plan:
  //
  //  0. Calculate source and destination domains.
  //  1. Build send and receive lists, and send messages.
  //  2. Evaluate local pieces directly.
  //  3. Receive messages and evaluate remaining pieces.
  //
  //===========================================================================
/*
#ifdef PRINT_DEBUG
  msg << "Starting BC Calculation for face "
      << getFace() << "." << endl;
#endif
*/
  //===========================================================================
  //  0. Calculate destination domain and the offset.
  //===========================================================================

  // Find the slab that is the destination. First get the whole
  // domain, including guard cells, and then restrict it to the area
  // that this BC will fill.

  const NDIndex<D>& domain(A.getDomain());

  NDIndex<D> src_slab = AddGuardCells(domain,A.getGuardCellSizes());

  // Direction Dim has faces 2*Dim and 2*Dim + 1, so the following
  // expression converts the face index to the direction index.

  unsigned d = this->getFace()/2;

  int offset;

  CalcParallelInterpolationDomain(A,*this,src_slab,offset);

  Domain_t dest_slab = src_slab;
  dest_slab[d] = dest_slab[d] + offset;

#ifdef PRINT_DEBUG
  msg << "dest_slab = " << dest_slab << endl;
  msg << "src_slab  = " << src_slab  << endl;
  //  stop_here();
#endif


  //===========================================================================
  //  1. Build send and receive lists and send messages
  //===========================================================================

  // Declare these at this scope so that we don't have to duplicate
  // the local code. (fillguardcells puts these in the scope of the
  // "if (nprocs > 1) { ... }" section, but has to duplicate the
  // code for the local fills as a result. The cost of this is a bit
  // of stackspace, and the cost of allocating an empty map.)

  // Container for holding Domain -> LField mapping
  // so that we can sort out which messages go where.

  typedef std::multimap<Domain_t,LField_t*, std::less<Domain_t> > ReceiveMap_t;

  // (Time this since it allocates an empty map.)



  ReceiveMap_t receive_map;



  // Number of nodes that will send us messages.

  int receive_count = 0;
  int send_count = 0;

  // Communications tag

  int bc_comm_tag;


  // Next fill the dest_list and src_list, lists of the LFields that
  // touch the destination and source domains, respectively.

  // (Do we need this for local-only code???)

  // (Also, if a domain ends up in both lists, it will only be
  // involved in local communication. We should structure this code to
  // take advantage of this, otherwise all existing parallel code is
  // going to incur additional overhead that really is unnecessary.)
  // (In other words, we should be able to do the general case, but
  // this capability shouldn't slow down the typical cases too much.)

  typedef std::vector<LField_t*> DestList_t;
  typedef std::vector<LField_t*> SrcList_t;
  typedef typename DestList_t::iterator DestListIterator_t;
  typedef typename SrcList_t::iterator SrcListIterator_t;

  DestList_t dest_list;
  SrcList_t src_list;

  dest_list.reserve(1);
  src_list.reserve(1);

  typename Field_t::iterator_if lf_i;

#ifdef PRINT_DEBUG
  msg << "Starting dest & src domain calculation." << endl;
#endif

  for (lf_i = A.begin_if(); lf_i != A.end_if(); ++lf_i)
    {
      LField_t &lf = *lf_i->second;

      // We fill if our OWNED domain touches the
      // destination slab.

      //const Domain_t &lf_allocated = lf.getAllocated();
      const Domain_t &lf_owned = lf.getOwned();

#ifdef PRINT_DEBUG
      msg << "  Processing subdomain : " << lf_owned << endl;
      //      stop_here();
#endif

      if (lf_owned.touches(dest_slab))
	dest_list.push_back(&lf);

      // We only provide data if our owned cells touch
      // the source slab (although we actually send the
      // allocated data).

      const Domain_t &lf_allocated = lf.getAllocated();

      if (lf_allocated.touches(src_slab))
	src_list.push_back(&lf);
    }

#ifdef PRINT_DEBUG
  msg << "  dest_list has " << dest_list.size() << " elements." << endl;
  msg << "  src_list has " << src_list.size() << " elements." << endl;
#endif

  DestListIterator_t dest_begin = dest_list.begin();
  DestListIterator_t dest_end   = dest_list.end();
  SrcListIterator_t src_begin  = src_list.begin();
  SrcListIterator_t src_end    = src_list.end();

  // Aliases to some of Field A's properties...

  const Layout_t &layout      = A.getLayout();
  const GuardCellSizes<D> &gc = A.getGuardCellSizes();

  int nprocs = Ippl::getNodes();

  if (nprocs > 1) // Skip send/receive code if we're single-processor.
    {


#ifdef PRINT_DEBUG
      msg << "Starting receive calculation." << endl;
      //      stop_here();
#endif

      //---------------------------------------------------
      // Receive calculation
      //---------------------------------------------------

      // Mask indicating the nodes will send us messages.

      std::vector<bool> receive_mask(nprocs,false);

      DestListIterator_t dest_i;

      for (dest_i = dest_begin; dest_i != dest_end; ++dest_i)
        {
          // Cache some information about this local array.

          LField_t &dest_lf = **dest_i;

          const Domain_t &dest_lf_alloc = dest_lf.getAllocated();

	  // Calculate the destination domain in this LField, and the
	  // source domain (which may be spread across multipple
	  // processors) from whence that domain will be filled:

	  const Domain_t dest_domain = dest_lf_alloc.intersect(dest_slab);

	  Domain_t src_domain = dest_domain;
	  //BENI:sign change for offset occurs when we iterate over destination first and calulate
	  // src domain from dest domain
	  src_domain[d] = src_domain[d] - offset;

          // Find the remote LFields that contain src_domain. Note
          // that we have to fill from the full allocated domains in
          // order to properly fill the corners of the boundary cells,
          // BUT we only need to intersect with the physical domain.
          // Intersecting the allocated domain would result in
          // unnecessary messages. (In fact, only the corners *need* to
          // send the allocated domains, but for regular decompositions,
          // sending the allocated domains will result in fewer
          // messages [albeit larger ones] than sending only from
          // physical cells.)

//BENI: include ghost cells for src_range
          typename Layout_t::touch_range_dv
            src_range(layout.touch_range_rdv(src_domain,gc));

	  // src_range is a begin/end pair into a list of remote
	  // domain/vnode pairs whose physical domains touch
	  // src_domain. Iterate through this list and set up the
	  // receive map and the receive mask.

          typename Layout_t::touch_iterator_dv rv_i;

          for (rv_i = src_range.first; rv_i != src_range.second; ++rv_i)
            {
              // Intersect src_domain with the allocated cells for the
	      // remote LField (rv_alloc). This will give us the strip
	      // that will be sent. Translate this domain back to the
	      // domain of the receiving LField.

	      //const Domain_t rv_alloc = AddGuardCells(rv_i->first,gc);
	      const Domain_t rv_alloc = rv_i->first;

              Domain_t hit = src_domain.intersect(rv_alloc);
			  //BENI: sign change
	      hit[d] = hit[d] + offset;

	      // Save this domain and the LField pointer

              typedef typename ReceiveMap_t::value_type value_type;

              receive_map.insert(value_type(hit,&dest_lf));

#ifdef PRINT_DEBUG
	      msg << "  Need remote data for domain: " << hit << endl;
#endif

              // Note who will be sending this data

              int rnode = rv_i->second->getNode();

              receive_mask[rnode] = true;

            } // rv_i
	} // dest_i

      receive_count = 0;

      for (int iproc = 0; iproc < nprocs; ++iproc)
	if (receive_mask[iproc]) ++receive_count;


#ifdef PRINT_DEBUG
      msg << "  Expecting to see " << receive_count << " messages." << endl;
      msg << "Done with receive calculation." << endl;
      //      stop_here();
#endif






#ifdef PRINT_DEBUG
      msg << "Starting send calculation" << endl;
#endif

      //---------------------------------------------------
      // Send calculation
      //---------------------------------------------------

      // Array of messages to be sent.

      std::vector<Message *> messages(nprocs);
      for (int miter=0; miter < nprocs; messages[miter++] = 0);

      // Debugging info.

#ifdef PRINT_DEBUG
      // KCC 3.2d has trouble with this. 3.3 doesn't, but
      // some are still using 3.2.
      //      vector<int> ndomains(nprocs,0);
      std::vector<int> ndomains(nprocs);
      for(int i = 0; i < nprocs; ++i) ndomains[i] = 0;
#endif

      SrcListIterator_t src_i;

      for (src_i = src_begin; src_i != src_end; ++src_i)
        {
          // Cache some information about this local array.

          LField_t &src_lf = **src_i;

	  // We need to send the allocated data to properly fill the
	  // corners when using periodic BCs in multipple dimensions.
	  // However, we don't want to send to nodes that only would
	  // receive data from our guard cells. Thus we do the
	  // intersection test with our owned data.

      const Domain_t &src_lf_owned = src_lf.getOwned();
	  const Domain_t &src_lf_alloc = src_lf.getAllocated();

	  // Calculate the owned and allocated parts of the source
	  // domain in this LField, and corresponding destination
	  // domains.

	  const Domain_t src_owned = src_lf_owned.intersect(src_slab);
	  const Domain_t src_alloc = src_lf_alloc.intersect(src_slab);

	  Domain_t dest_owned = src_owned;
	  dest_owned[d] = dest_owned[d] + offset;

	  Domain_t dest_alloc = src_alloc;
	  dest_alloc[d] = dest_alloc[d] + offset;

#ifdef PRINT_DEBUG
	  msg << "  Considering LField with the domains:" << endl;
	  msg << "     owned = " << src_lf_owned << endl;
	  msg << "     alloc = " << src_lf_alloc << endl;
	  msg << "  The intersections with src_slab are:" << endl;
	  msg << "     owned = " << src_owned << endl;
	  msg << "     alloc = " << src_alloc << endl;
#endif

          // Find the remote LFields whose allocated cells (note the
	  // additional "gc" arg) are touched by dest_owned.

          typename Layout_t::touch_range_dv
            dest_range(layout.touch_range_rdv(dest_owned,gc));

          typename Layout_t::touch_iterator_dv rv_i;
/*
#ifdef PRINT_DEBUG
	  msg << "  Touch calculation found "
	      << distance(dest_range.first, dest_range.second)
	      << " elements." << endl;
#endif
*/

          for (rv_i = dest_range.first; rv_i != dest_range.second; ++rv_i)
            {
              // Find the intersection of the returned domain with the
	      // allocated version of the translated source domain.
	      // Translate this intersection back to the source side.

              Domain_t hit = dest_alloc.intersect(rv_i->first);
	      hit[d] = hit[d] - offset;

              // Find out who owns this remote domain.

              int rnode = rv_i->second->getNode();

#ifdef PRINT_DEBUG
              msg << "  Overlap domain = " << rv_i->first << endl;
              msg << "  Inters. domain = " << hit;
              msg << "  --> node " << rnode << endl;
#endif

              // Create an LField iterator for this intersection region,
              // and try to compress it. (Copied from fillGuardCells -
	      // not quite sure how this works yet. JAC)

              // storage for LField compression

              Element_t compressed_value;
              LFI_t msgval = src_lf.begin(hit, compressed_value);
              msgval.TryCompress();

              // Put intersection domain and field data into message

              if (!messages[rnode])
		{
		  messages[rnode] = new Message;
		  PAssert(messages[rnode]);
		}

              messages[rnode]->put(hit);    // puts Dim items in Message
              messages[rnode]->put(msgval); // puts 3 items in Message

#ifdef PRINT_DEBUG
              ndomains[rnode]++;
#endif
            }  // rv_i
	} // src_i

      // Get message tag.

      bc_comm_tag =
	Ippl::Comm->next_tag(BC_PARALLEL_PERIODIC_TAG,BC_TAG_CYCLE);



      // Send the messages.

      for (int iproc = 0; iproc < nprocs; ++iproc)
	{
	  if (messages[iproc])
	    {

#ifdef PRINT_DEBUG
	      msg << "  ParallelPeriodicBCApply: Sending message to node "
		  << iproc << endl
		  << "    number of domains  = " << ndomains[iproc] << endl
		  << "    number of MsgItems = "
		  << messages[iproc]->size() << endl;
#endif

	      Ippl::Comm->send(messages[iproc], iproc, bc_comm_tag);
	      ++send_count;

	    }

	}

#ifdef PRINT_DEBUG
      msg << "  Sent " << send_count << " messages" << endl;
      msg << "Done with send." << endl;
#endif





    } // if (nprocs > 1)




  //===========================================================================
  //  2. Evaluate local pieces directly.
  //===========================================================================

#ifdef PRINT_DEBUG
  msg << "Starting local calculation." << endl;
#endif

  DestListIterator_t dest_i;

  for (dest_i = dest_begin; dest_i != dest_end; ++dest_i)
    {
      // Cache some information about this local array.

      LField_t &dest_lf = **dest_i;

      const Domain_t &dest_lf_alloc = dest_lf.getAllocated();
      //const Domain_t &dest_lf_owned = dest_lf.getOwned();

      const Domain_t dest_domain = dest_lf_alloc.intersect(dest_slab);

      Domain_t src_domain = dest_domain;
	  //BENI:sign change for offset occurs when we iterate over destination first and calulate
	  // src domain from dest domain
      src_domain[d] = src_domain[d] - offset;

      SrcListIterator_t src_i;

      for (src_i = src_begin; src_i != src_end; ++src_i)
        {
          // Cache some information about this local array.

          LField_t &src_lf = **src_i;

	  // Unlike fillGuardCells, we need to send the allocated
	  // data.  This is necessary to properly fill the corners
	  // when using periodic BCs in multipple dimensions.

          //const Domain_t &src_lf_owned = src_lf.getOwned();
          const Domain_t &src_lf_alloc = src_lf.getAllocated();

	  // Only fill from LFields whose physical domain touches
	  // the translated destination domain.

	  if (src_domain.touches(src_lf_alloc))
	    {
	      // Worry about compression. Should do this right
	      // (considering the four different combinations), but
	      // for now just do what the serial version does:

	      dest_lf.Uncompress();

	      // Calculate the actual source and destination domains.

	      Domain_t real_src_domain =
		src_domain.intersect(src_lf_alloc);

	      Domain_t real_dest_domain = real_src_domain;
		  //BENI: same sign change as above
	      real_dest_domain[d] = real_dest_domain[d] + offset;

#ifdef PRINT_DEBUG
	      msg << "  Copying local data . . ." << endl;
	      msg << "    source domain = " << real_src_domain << endl;
	      msg << "    dest domain   = " << real_dest_domain << endl;
#endif

	      // Build the iterators for the copy

	      LFI_t lhs = dest_lf.begin(real_dest_domain);
	      LFI_t rhs = src_lf.begin(real_src_domain);

	      // And do the assignment:

	      if (this->getComponent() == BCondBase<T,D,M,C>::allComponents)
		{
		  BrickExpression<D,LFI_t,LFI_t,OpInterpolation<T> >
		    (lhs,rhs,OpInterpolation<T>()).apply();
		}
	      else
		{
		  BrickExpression<D,LFI_t,LFI_t,OpInterpolationComponent<T> >
		    (lhs,rhs,OpInterpolationComponent<T>(this->getComponent())).apply();
		}
            }
        }
    }

#ifdef PRINT_DEBUG
  msg << "Done with local calculation." << endl;
#endif




  //===========================================================================
  //  3. Receive messages and evaluate remaining pieces.
  //===========================================================================

  if (nprocs > 1)
    {



#ifdef PRINT_DEBUG
      msg << "Starting receive..." << endl;
      //      stop_here();
#endif

      while (receive_count > 0)
	{

	  // Receive the next message.

	  int any_node = COMM_ANY_NODE;



	  Message* message =
	    Ippl::Comm->receive_block(any_node,bc_comm_tag);
	  PAssert(message);

	  --receive_count;



	  // Determine the number of domains being sent

	  int ndomains = message->size() / (D + 3);

#ifdef PRINT_DEBUG
	  msg << "  Message received from node "
	      << any_node << "," << endl
	      << "  number of domains = " << ndomains << endl;
#endif

	  for (int idomain=0; idomain < ndomains; ++idomain)
	    {
	      // Extract the intersection domain from the message
	      // and translate it to the destination side.

	      Domain_t intersection;
	      intersection.getMessage(*message);
		  //BENI:: sign change
	      intersection[d] = intersection[d] + offset;

	      // Extract the rhs iterator from it.

	      Element_t compressed_value;
	      LFI_t rhs(compressed_value);
	      rhs.getMessage(*message);

#ifdef PRINT_DEBUG
	      msg << "  Received remote overlap region = "
		  << intersection << endl;
#endif

	      // Find the LField it is destined for.

	      typename ReceiveMap_t::iterator hit =
		receive_map.find(intersection);

	      PAssert(hit != receive_map.end());

	      // Build the lhs brick iterator.

	      // Should have been
	      // LField<T,D> &lf = *hit->second;
	      // but SGI's 7.2  multimap doesn't have op->().

	      LField<T,D> &lf = *(*hit).second;

	      // Check and see if we really have to do this.

#ifdef PRINT_DEBUG
	      msg << "   LHS compressed ? " << lf.IsCompressed();
	      msg << ", LHS value = " << *lf.begin() << endl;
	      msg << "   RHS compressed ? " << rhs.IsCompressed();
	      msg << ", RHS value = " << *rhs << endl;
	      msg << "   *rhs == *lf.begin() ? "
		  << (*rhs == *lf.begin()) << endl;
#endif
	      if (!(rhs.IsCompressed() && lf.IsCompressed() &&
		    (*rhs == *lf.begin())))
		{
		  // Yep. gotta do it.

		  lf.Uncompress();
		  LFI_t lhs = lf.begin(intersection);

		  // Do the assignment.

#ifdef PRINT_DEBUG
		  msg << "   Doing copy." << endl;
#endif

		  //BrickExpression<D,LFI_t,LFI_t,OpAssign>(lhs,rhs).apply();
		  BrickExpression<D,LFI_t,LFI_t,OpInterpolation<T> >(lhs,rhs,OpInterpolation<T>()).apply();
		}

	      // Take that entry out of the receive list.

	      receive_map.erase(hit);
	    }

	  delete message;
	}


#ifdef PRINT_DEBUG
      msg << "Done with receive." << endl;
#endif

    }
}



//////////////////////////////////////////////////////////////////////

// Applicative templates for ExtrapolateFace:

// Standard, for applying to all components of elemental type:
template<class T>
struct OpExtrapolate
{
  OpExtrapolate(const T& o, const T& s) : Offset(o), Slope(s) {}
  T Offset, Slope;
};
template<class T>
inline void PETE_apply(const OpExtrapolate<T>& e, T& a, const T& b)
{ a = b*e.Slope + e.Offset; }

// Special, for applying to single component of multicomponent elemental type:
template<class T>
struct OpExtrapolateComponent
{
  OpExtrapolateComponent(const T& o, const T& s, int c)
	 : Offset(o), Slope(s), Component(c) {}
  T Offset, Slope;
  int Component;
};
template<class T>
inline void PETE_apply(const OpExtrapolateComponent<T>& e, T& a, const T& b)
{
  a[e.Component] = b[e.Component]*e.Slope[e.Component] + e.Offset[e.Component];
}

// Following specializations are necessary because of the runtime branches in
// functions like these in code below:
// 		  if (ef.Component == BCondBase<T,D,M,Cell>::allComponents) {
// 		    BrickExpression<D,LFI,LFI,OpExtrapolate<T> >
// 		      (lhs,rhs,OpExtrapolate<T>(ef.Offset,ef.Slope)).apply();
// 		  } else {
// 		    BrickExpression<D,LFI,LFI,OpExtrapolateComponent<T> >
// 		      (lhs,rhs,OpExtrapolateComponent<T>
// 		       (ef.Offset,ef.Slope,ef.Component)).apply();
// 		  }
// which unfortunately force instantiation of OpExtrapolateComponent instances
// for non-multicomponent types like {char,double,...}. Note: if user uses
// non-multicomponent (no operator[]) types of his own, he'll get a compile
// error.

COMPONENT_APPLY_BUILTIN(OpExtrapolateComponent,char)
COMPONENT_APPLY_BUILTIN(OpExtrapolateComponent,bool)
COMPONENT_APPLY_BUILTIN(OpExtrapolateComponent,int)
COMPONENT_APPLY_BUILTIN(OpExtrapolateComponent,unsigned)
COMPONENT_APPLY_BUILTIN(OpExtrapolateComponent,short)
COMPONENT_APPLY_BUILTIN(OpExtrapolateComponent,long)
COMPONENT_APPLY_BUILTIN(OpExtrapolateComponent,float)
COMPONENT_APPLY_BUILTIN(OpExtrapolateComponent,double)
COMPONENT_APPLY_BUILTIN(OpExtrapolateComponent,std::complex<double>)

//////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// For unspecified centering, can't implement ExtrapolateFace::apply()
// correctly, and can't partial-specialize yet, so... don't have a prototype
// for unspecified centering, so user gets a compile error if he tries to
// invoke it for a centering not yet implemented. Implement external functions
// which are specializations for the various centerings
// {Cell,Vert,CartesianCentering}; these are called from the general
// ExtrapolateFace::apply() function body.
//----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////

template<class T, unsigned D, class M>
void ExtrapolateFaceBCApply(ExtrapolateFace<T,D,M,Cell>& ef,
			    Field<T,D,M,Cell>& A );
template<class T, unsigned D, class M>
void ExtrapolateFaceBCApply(ExtrapolateFace<T,D,M,Vert>& ef,
			    Field<T,D,M,Vert>& A );
template<class T, unsigned D, class M>
void ExtrapolateFaceBCApply(ExtrapolateFace<T,D,M,Edge>& ef,
			    Field<T,D,M,Edge>& A );
template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
void ExtrapolateFaceBCApply(ExtrapolateFace<T,D,M,
			    CartesianCentering<CE,D,NC> >& ef,
			    Field<T,D,M,CartesianCentering<CE,D,NC> >& A );

template<class T, unsigned D, class M, class C>
void ExtrapolateFace<T,D,M,C>::apply( Field<T,D,M,C>& A )
{
  ExtrapolateFaceBCApply(*this, A);
}


template<class T, unsigned D, class M, class C>
inline void
ExtrapolateFaceBCApply2(const NDIndex<D> &dest, const NDIndex<D> &src,
  LField<T,D> &fill, LField<T,D> &from, const NDIndex<D> &from_alloc,
  ExtrapolateFace<T,D,M,C> &ef)
{
  // If both the 'fill' and 'from' are compressed and applying the boundary
  // condition on the compressed value would result in no change to
  // 'fill' we don't need to uncompress.

  if (fill.IsCompressed() && from.IsCompressed())
    {
      // So far, so good. Let's check to see if the boundary condition
      // would result in filling the guard cells with a different value.

      T a = fill.getCompressedData(), aref = a;
      T b = from.getCompressedData();
      if (ef.getComponent() == BCondBase<T,D,M,C>::allComponents)
	{
	  OpExtrapolate<T> op(ef.getOffset(),ef.getSlope());
	  PETE_apply(op, a, b);
	}
      else
	{
          int d = (ef.getComponent() % D + D) % D;
          OpExtrapolateComponent<T>
            op(ef.getOffset(),ef.getSlope(),d);
          PETE_apply(op, a, b);
	}
      if (a == aref)
	{
	  // Yea! We're outta here.

	  return;
	}
    }

  // Well poop, we have no alternative but to uncompress the region
  // we're filling.

  fill.Uncompress();

  NDIndex<D> from_it = src.intersect(from_alloc);
  NDIndex<D> fill_it = dest.plugBase(from_it);

  // Build iterators for the copy...

  typedef typename LField<T,D>::iterator LFI;
  LFI lhs = fill.begin(fill_it);
  LFI rhs = from.begin(from_it);

  // And do the assignment.

  if (ef.getComponent() == BCondBase<T,D,M,C>::allComponents)
    {
      BrickExpression<D,LFI,LFI,OpExtrapolate<T> >
	(lhs,rhs,OpExtrapolate<T>(ef.getOffset(),ef.getSlope())).apply();
    }
  else
    {
      BrickExpression<D,LFI,LFI,OpExtrapolateComponent<T> >
	(lhs,rhs,OpExtrapolateComponent<T>
	 (ef.getOffset(),ef.getSlope(),ef.getComponent())).apply();
    }
}


//-----------------------------------------------------------------------------
// Specialization of ExtrapolateFace::apply() for Cell centering.
// Rather, indirectly-called specialized global function ExtrapolateFaceBCApply
//-----------------------------------------------------------------------------

template<class T, unsigned D, class M>
void ExtrapolateFaceBCApply(ExtrapolateFace<T,D,M,Cell>& ef,
			    Field<T,D,M,Cell>& A )
{



  // Find the slab that is the destination.
  // That is, in English, get an NDIndex spanning elements in the guard layers
  // on the face associated with the ExtrapaloteFace object:

  const NDIndex<D>& domain( A.getDomain() ); // Spans whole Field
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());

  // The direction (dimension of the Field) associated with the active face.
  // The numbering convention makes this division by two return the right
  // value, which will be between 0 and (D-1):

  unsigned d = ef.getFace()/2;
  int offset;

  // The following bitwise AND logical test returns true if ef.m_face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if ef.m_face is even (meaning the "low" or "left" face in
  // the numbering convention):

  if (ef.getFace() & 1)
    {
      // For "high" face, index in active direction goes from max index of
      // Field plus 1 to the same plus number of guard layers:
      // TJW: this used to say "leftGuard(d)", which I think was wrong:

      slab[d] = Index( domain[d].max() + 1, domain[d].max() + A.rightGuard(d));

      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:

      offset = 2*domain[d].max() + 1;
    }
  else
    {
      // For "low" face, index in active direction goes from min index of
      // Field minus the number of guard layers (usually a negative number)
      // to the same min index minus 1 (usually negative, and usually -1):

      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );

      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:

      offset = 2*domain[d].min() - 1;
    }

  // Loop over all the LField's in the Field A:

  typename Field<T,D,M,Cell>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      // Pointer to the data for the current LField (right????):

      LField<T,D> &fill = *(*fill_i).second;

      // NDIndex spanning all elements in the LField, including the guards:

      const NDIndex<D> &fill_alloc = fill.getAllocated();

      // If the previously-created boundary guard-layer NDIndex "slab"
      // contains any of the elements in this LField (they will be guard
      // elements if it does), assign the values into them here by applying
      // the boundary condition:

      if (slab.touches(fill_alloc))
        {
          // Find what it touches in this LField.

          NDIndex<D> dest = slab.intersect(fill_alloc);

          // For extrapolation boundary conditions, the boundary guard-layer
	  // elements are typically copied from interior values; the "src"
	  // NDIndex specifies the interior elements to be copied into the
	  // "dest" boundary guard-layer elements (possibly after some
	  // mathematical operations like multipplying by minus 1 later):

          NDIndex<D> src = dest;

	  // Now calculate the interior elements; the offset variable computed
	  // above makes this correct for "low" or "high" face cases:

          src[d] = offset - src[d];

	  // At this point, we need to see if 'src' is fully contained by
	  // by 'fill_alloc'. If it is, we have a lot less work to do.

	  if (fill_alloc.contains(src))
	    {
	      // Great! Our domain contains the elements we're filling from.

	      ExtrapolateFaceBCApply2(dest, src, fill, fill,
	        fill_alloc, ef);
	    }
	  else
	    {
	      // Yuck! Our domain doesn't contain all of the src. We
	      // must loop over LFields to find the ones the touch the src.

	      typename Field<T,D,M,Cell>::iterator_if from_i;
	      for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
		{
		  // Cache a few things.

		  LField<T,D> &from = *(*from_i).second;
		  const NDIndex<D> &from_owned = from.getOwned();
		  const NDIndex<D> &from_alloc = from.getAllocated();

		  // If src touches this LField...

		  if (src.touches(from_owned))
		    ExtrapolateFaceBCApply2(dest, src, fill, from,
		      from_alloc, ef);
		}
	    }
	}
    }
}


//-----------------------------------------------------------------------------
// Specialization of ExtrapolateFace::apply() for Vert centering.
// Rather, indirectly-called specialized global function ExtrapolateFaceBCApply
//-----------------------------------------------------------------------------

template<class T, unsigned D, class M>
void ExtrapolateFaceBCApply(ExtrapolateFace<T,D,M,Vert>& ef,
			    Field<T,D,M,Vert>& A )
{



  // Find the slab that is the destination.
  // That is, in English, get an NDIndex spanning elements in the guard layers
  // on the face associated with the ExtrapaloteFace object:

  const NDIndex<D>& domain(A.getDomain());
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());

  // The direction (dimension of the Field) associated with the active face.
  // The numbering convention makes this division by two return the right
  // value, which will be between 0 and (D-1):

  unsigned d = ef.getFace()/2;
  int offset;

  // The following bitwise AND logical test returns true if ef.m_face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if ef.m_face is even (meaning the "low" or "left" face
  // in the numbering convention):

  if ( ef.getFace() & 1 )
    {
      // For "high" face, index in active direction goes from max index of
      // Field plus 1 to the same plus number of guard layers:
      // TJW: this used to say "leftGuard(d)", which I think was wrong:

      slab[d] = Index( domain[d].max() + 1, domain[d].max() + A.rightGuard(d));

      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:
      // N.B.: the extra -1 here is what distinguishes this Vert-centered
      // implementation from the Cell-centered one:

      offset = 2*domain[d].max() + 1 - 1;
    }
  else
    {
      // For "low" face, index in active direction goes from min index of
      // Field minus the number of guard layers (usually a negative number)
      // to the same min index minus 1 (usually negative, and usually -1):

      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:
      // N.B.: the extra +1 here is what distinguishes this Vert-centered
      // implementation from the Cell-centered one:

      offset = 2*domain[d].min() - 1 + 1;
    }

  // Loop over all the LField's in the Field A:

  typename Field<T,D,M,Vert>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      // Pointer to the data for the current LField (right????):

      LField<T,D> &fill = *(*fill_i).second;
      // NDIndex spanning all elements in the LField, including the guards:

      const NDIndex<D> &fill_alloc = fill.getAllocated();
      // If the previously-created boundary guard-layer NDIndex "slab"
      // contains any of the elements in this LField (they will be guard
      // elements if it does), assign the values into them here by applying
      // the boundary condition:

      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.

          NDIndex<D> dest = slab.intersect( fill_alloc );

          // For exrapolation boundary conditions, the boundary guard-layer
	  // elements are typically copied from interior values; the "src"
	  // NDIndex specifies the interior elements to be copied into the
	  // "dest" boundary guard-layer elements (possibly after some
	  // mathematical operations like multipplying by minus 1 later):

          NDIndex<D> src = dest;

	  // Now calculate the interior elements; the offset variable computed
	  // above makes this correct for "low" or "high" face cases:

          src[d] = offset - src[d];

	  // At this point, we need to see if 'src' is fully contained by
	  // by 'fill_alloc'. If it is, we have a lot less work to do.

	  if (fill_alloc.contains(src))
	    {
	      // Great! Our domain contains the elements we're filling from.

	      ExtrapolateFaceBCApply2(dest, src, fill, fill,
	        fill_alloc, ef);
	    }
	  else
	    {
	      // Yuck! Our domain doesn't contain all of the src. We
	      // must loop over LFields to find the ones the touch the src.

	      typename Field<T,D,M,Vert>::iterator_if from_i;
	      for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
		{
		  // Cache a few things.

		  LField<T,D> &from = *(*from_i).second;
		  const NDIndex<D> &from_owned = from.getOwned();
		  const NDIndex<D> &from_alloc = from.getAllocated();

		  // If src touches this LField...

		  if (src.touches(from_owned))
		    ExtrapolateFaceBCApply2(dest, src, fill, from,
		      from_alloc, ef);
		}
	    }
	}
    }
}



//-----------------------------------------------------------------------------
// Specialization of ExtrapolateFace::apply() for Edge centering.
// Rather, indirectly-called specialized global function ExtrapolateFaceBCApply
//-----------------------------------------------------------------------------

template<class T, unsigned D, class M>
void ExtrapolateFaceBCApply(ExtrapolateFace<T,D,M,Edge>& ef,
			    Field<T,D,M,Edge>& A )
{
  // Find the slab that is the destination.
  // That is, in English, get an NDIndex spanning elements in the guard layers
  // on the face associated with the ExtrapaloteFace object:

  const NDIndex<D>& domain(A.getDomain());
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());

  // The direction (dimension of the Field) associated with the active face.
  // The numbering convention makes this division by two return the right
  // value, which will be between 0 and (D-1):

  unsigned d = ef.getFace()/2;
  int offset;

  // The following bitwise AND logical test returns true if ef.m_face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if ef.m_face is even (meaning the "low" or "left" face
  // in the numbering convention):

  if ( ef.getFace() & 1 )
    {
      // For "high" face, index in active direction goes from max index of
      // Field plus 1 to the same plus number of guard layers:
      // TJW: this used to say "leftGuard(d)", which I think was wrong:

      slab[d] = Index( domain[d].max() + 1, domain[d].max() + A.rightGuard(d));

      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:
      // N.B.: the extra -1 here is what distinguishes this Edge-centered
      // implementation from the Cell-centered one:

      offset = 2*domain[d].max() + 1 - 1;
    }
  else
    {
      // For "low" face, index in active direction goes from min index of
      // Field minus the number of guard layers (usually a negative number)
      // to the same min index minus 1 (usually negative, and usually -1):

      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:
      // N.B.: the extra +1 here is what distinguishes this Edge-centered
      // implementation from the Cell-centered one:

      offset = 2*domain[d].min() - 1 + 1;
    }

  // Loop over all the LField's in the Field A:

  typename Field<T,D,M,Edge>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      // Pointer to the data for the current LField (right????):

      LField<T,D> &fill = *(*fill_i).second;
      // NDIndex spanning all elements in the LField, including the guards:

      const NDIndex<D> &fill_alloc = fill.getAllocated();
      // If the previously-created boundary guard-layer NDIndex "slab"
      // contains any of the elements in this LField (they will be guard
      // elements if it does), assign the values into them here by applying
      // the boundary condition:

      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.

          NDIndex<D> dest = slab.intersect( fill_alloc );

          // For exrapolation boundary conditions, the boundary guard-layer
	  // elements are typically copied from interior values; the "src"
	  // NDIndex specifies the interior elements to be copied into the
	  // "dest" boundary guard-layer elements (possibly after some
	  // mathematical operations like multipplying by minus 1 later):

          NDIndex<D> src = dest;

	  // Now calculate the interior elements; the offset variable computed
	  // above makes this correct for "low" or "high" face cases:

          src[d] = offset - src[d];

	  // At this point, we need to see if 'src' is fully contained by
	  // by 'fill_alloc'. If it is, we have a lot less work to do.

	  if (fill_alloc.contains(src))
	    {
	      // Great! Our domain contains the elements we're filling from.

	      ExtrapolateFaceBCApply2(dest, src, fill, fill,
	        fill_alloc, ef);
	    }
	  else
	    {
	      // Yuck! Our domain doesn't contain all of the src. We
	      // must loop over LFields to find the ones the touch the src.

	      typename Field<T,D,M,Edge>::iterator_if from_i;
	      for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
		{
		  // Cache a few things.

		  LField<T,D> &from = *(*from_i).second;
		  const NDIndex<D> &from_owned = from.getOwned();
		  const NDIndex<D> &from_alloc = from.getAllocated();

		  // If src touches this LField...

		  if (src.touches(from_owned))
		    ExtrapolateFaceBCApply2(dest, src, fill, from,
		      from_alloc, ef);
		}
	    }
	}
    }
}


//-----------------------------------------------------------------------------
// Specialization of ExtrapolateFace::apply() for CartesianCentering centering.
// Rather,indirectly-called specialized global function ExtrapolateFaceBCApply:
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
void ExtrapolateFaceBCApply(ExtrapolateFace<T,D,M,
			    CartesianCentering<CE,D,NC> >& ef,
			    Field<T,D,M,CartesianCentering<CE,D,NC> >& A )
{



  // Find the slab that is the destination.
  // That is, in English, get an NDIndex spanning elements in the guard layers
  // on the face associated with the ExtrapaloteFace object:

  const NDIndex<D>& domain( A.getDomain() ); // Spans whole Field
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());

  // The direction (dimension of the Field) associated with the active face.
  // The numbering convention makes this division by two return the right
  // value, which will be between 0 and (D-1):

  unsigned d = ef.getFace()/2;
  int offset;

  // The following bitwise AND logical test returns true if ef.m_face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if ef.m_face is even (meaning the "low" or "left" face
  // in the numbering convention):

  if ( ef.getFace() & 1 )
    {
      // offset is used in computing interior elements used in computing fill
      // values for boundary guard  elements; see below:
      // Do the right thing for CELL or VERT centering for this component (or
      // all components, if the PeriodicFace object so specifies):

      if (ef.getComponent() == BCondBase<T,D,M,CartesianCentering<CE,D,NC> >::
	  allComponents)
	{
	  // Make sure all components are really centered the same, as assumed:

	  CenteringEnum centering0 = CE[0 + d*NC]; // 1st component along dir d
	  for (unsigned int c=1; c<NC; c++)
	    {
	      // Compare other components with 1st
	      if (CE[c + d*NC] != centering0)
		ERRORMSG("ExtrapolateFaceBCApply: BCond thinks all components"
			 << " have same centering along direction " << d
			 << ", but it isn't so." << endl);
	    }

	  // Now do the right thing for CELL or VERT centering of
	  // all components:

	  // For "high" face, index in active direction goes from max index of
	  // Field plus 1 to the same plus number of guard layers:

	  slab[d] = Index(domain[d].max() + 1,
			  domain[d].max() + A.rightGuard(d));

	  if (centering0 == CELL)
	    {
	      offset = 2*domain[d].max() + 1 ;    // CELL case
	    }
	  else
	    {
	      offset = 2*domain[d].max() + 1 - 1; // VERT case
	    }
	}
      else
	{
	  // The BC applies only to one component, not all:
	  // Do the right thing for CELL or VERT centering of the component:
	  if (CE[ef.getComponent() + d*NC] == CELL)
	    {
	      // For "high" face, index in active direction goes from max index
	      // of cells in the Field plus 1 to the same plus number of guard
	      // layers:
	      int highcell = A.get_mesh().gridSizes[d] - 2;
	      slab[d] = Index(highcell + 1, highcell + A.rightGuard(d));

	      //	      offset = 2*domain[d].max() + 1 ;    // CELL case
	      offset = 2*highcell + 1 ;    // CELL case
	    }
	  else
	    {
	      // For "high" face, index in active direction goes from max index
	      // of verts in the Field plus 1 to the same plus number of guard
	      // layers:
	      int highvert = A.get_mesh().gridSizes[d] - 1;
	      slab[d] = Index(highvert + 1, highvert + A.rightGuard(d));

	      //	      offset = 2*domain[d].max() + 1 - 1; // VERT case
	      offset = 2*highvert + 1 - 1; // VERT case
	    }
	}
    }
  else
    {
      // For "low" face, index in active direction goes from min index of
      // Field minus the number of guard layers (usually a negative number)
      // to the same min index minus 1 (usually negative, and usually -1):

      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );

      // offset is used in computing interior elements used in computing fill
      // values for boundary guard  elements; see below:
      // Do the right thing for CELL or VERT centering for this component (or
      // all components, if the PeriodicFace object so specifies):

      if (ef.getComponent() == BCondBase<T,D,M,CartesianCentering<CE,D,NC> >::
	  allComponents)
	{
	  // Make sure all components are really centered the same, as assumed:

	  CenteringEnum centering0 = CE[0 + d*NC]; // 1st component along dir d
	  for (unsigned int c=1; c<NC; c++)
	    {
	      // Compare other components with 1st

	      if (CE[c + d*NC] != centering0)
		ERRORMSG("ExtrapolateFaceBCApply: BCond thinks all components"
		     << " have same centering along direction " << d
		     << ", but it isn't so." << endl);
	    }

	  // Now do the right thing for CELL or VERT centering of all
	  // components:

	  if (centering0 == CELL)
	    {
	      offset = 2*domain[d].min() - 1;     // CELL case
	    }
	  else
	    {
	      offset = 2*domain[d].min() - 1 + 1; // VERT case
	    }
	}
      else
	{
	  // The BC applies only to one component, not all:
	  // Do the right thing for CELL or VERT centering of the component:

	  if (CE[ef.getComponent() + d*NC] == CELL)
	    {
	      offset = 2*domain[d].min() - 1;     // CELL case
	    }
	  else
	    {
	      offset = 2*domain[d].min() - 1 + 1; // VERT case
	    }
	}
    }

  // Loop over all the LField's in the Field A:

  typename Field<T,D,M,CartesianCentering<CE,D,NC> >::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      // Pointer to the data for the current LField (right????):

      LField<T,D> &fill = *(*fill_i).second;

      // NDIndex spanning all elements in the LField, including the guards:

      const NDIndex<D> &fill_alloc = fill.getAllocated();

      // If the previously-created boundary guard-layer NDIndex "slab"
      // contains any of the elements in this LField (they will be guard
      // elements if it does), assign the values into them here by applying
      // the boundary condition:

      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.

          NDIndex<D> dest = slab.intersect( fill_alloc );

          // For exrapolation boundary conditions, the boundary guard-layer
	  // elements are typically copied from interior values; the "src"
	  // NDIndex specifies the interior elements to be copied into the
	  // "dest" boundary guard-layer elements (possibly after some
	  // mathematical operations like multipplying by minus 1 later):

          NDIndex<D> src = dest;

	  // Now calculate the interior elements; the offset variable computed
	  // above makes this correct for "low" or "high" face cases:

          src[d] = offset - src[d];

	  // At this point, we need to see if 'src' is fully contained by
	  // by 'fill_alloc'. If it is, we have a lot less work to do.

	  if (fill_alloc.contains(src))
	    {
	      // Great! Our domain contains the elements we're filling from.

	      ExtrapolateFaceBCApply2(dest, src, fill, fill,
	        fill_alloc, ef);
	    }
	  else
	    {
	      // Yuck! Our domain doesn't contain all of the src. We
	      // must loop over LFields to find the ones the touch the src.

	      typename Field<T,D,M,CartesianCentering<CE,D,NC> >::iterator_if
		from_i;
	      for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
		{
		  // Cache a few things.

		  LField<T,D> &from = *(*from_i).second;
		  const NDIndex<D> &from_owned = from.getOwned();
		  const NDIndex<D> &from_alloc = from.getAllocated();

		  // If src touches this LField...

		  if (src.touches(from_owned))
		    ExtrapolateFaceBCApply2(dest, src, fill, from,
		      from_alloc, ef);
		}
	    }
	}
    }
}

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// TJW added 12/16/1997 as per Tecolote Team's request... BEGIN
//////////////////////////////////////////////////////////////////////

// Applicative templates for ExtrapolateAndZeroFace:

// Standard, for applying to all components of elemental type:
template<class T>
struct OpExtrapolateAndZero
{
  OpExtrapolateAndZero(const T& o, const T& s) : Offset(o), Slope(s) {}
  T Offset, Slope;
};
template<class T>
inline void PETE_apply(const OpExtrapolateAndZero<T>& e, T& a, const T& b)
{ a = b*e.Slope + e.Offset; }

// Special, for applying to single component of multicomponent elemental type:
template<class T>
struct OpExtrapolateAndZeroComponent
{
  OpExtrapolateAndZeroComponent(const T& o, const T& s, int c)
	 : Offset(o), Slope(s), Component(c) {}
  T Offset, Slope;
  int Component;
};
template<class T>
inline void PETE_apply(const OpExtrapolateAndZeroComponent<T>& e, T& a,
                       const T& b)
{
  a[e.Component] = b[e.Component]*e.Slope[e.Component] + e.Offset[e.Component];
}

// Following specializations are necessary because of the runtime branches in
// functions like these in code below:
// 		  if (ef.Component == BCondBase<T,D,M,Cell>::allComponents) {
// 		    BrickExpression<D,LFI,LFI,OpExtrapolateAndZero<T> >
// 		      (lhs,rhs,OpExtrapolateAndZero<T>(ef.Offset,ef.Slope)).
//                    apply();
// 		  } else {
// 		    BrickExpression<D,LFI,LFI,
//                    OpExtrapolateAndZeroComponent<T> >
// 		      (lhs,rhs,OpExtrapolateAndZeroComponent<T>
// 		       (ef.Offset,ef.Slope,ef.Component)).apply();
// 		  }
// which unfortunately force instantiation of OpExtrapolateAndZeroComponent
// instances for non-multicomponent types like {char,double,...}. Note: if
// user uses non-multicomponent (no operator[]) types of his own, he'll get a
// compile error.

COMPONENT_APPLY_BUILTIN(OpExtrapolateAndZeroComponent,char)
COMPONENT_APPLY_BUILTIN(OpExtrapolateAndZeroComponent,bool)
COMPONENT_APPLY_BUILTIN(OpExtrapolateAndZeroComponent,int)
COMPONENT_APPLY_BUILTIN(OpExtrapolateAndZeroComponent,unsigned)
COMPONENT_APPLY_BUILTIN(OpExtrapolateAndZeroComponent,short)
COMPONENT_APPLY_BUILTIN(OpExtrapolateAndZeroComponent,long)
COMPONENT_APPLY_BUILTIN(OpExtrapolateAndZeroComponent,float)
COMPONENT_APPLY_BUILTIN(OpExtrapolateAndZeroComponent,double)
COMPONENT_APPLY_BUILTIN(OpExtrapolateAndZeroComponent,std::complex<double>)

// Special, for assigning to single component of multicomponent elemental type:
template<class T>
struct OpAssignComponent
{
  OpAssignComponent(int c)
    : Component(c) { }
  int Component;
};

template<class T, class T1>
inline void PETE_apply(const OpAssignComponent<T>& e, T& a, const T1& b)
{
  a[e.Component] = b;
}

COMPONENT_APPLY_BUILTIN(OpAssignComponent,char)
COMPONENT_APPLY_BUILTIN(OpAssignComponent,bool)
COMPONENT_APPLY_BUILTIN(OpAssignComponent,int)
COMPONENT_APPLY_BUILTIN(OpAssignComponent,unsigned)
COMPONENT_APPLY_BUILTIN(OpAssignComponent,short)
COMPONENT_APPLY_BUILTIN(OpAssignComponent,long)
COMPONENT_APPLY_BUILTIN(OpAssignComponent,float)
COMPONENT_APPLY_BUILTIN(OpAssignComponent,double)
COMPONENT_APPLY_BUILTIN(OpAssignComponent,std::complex<double>)

//////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// For unspecified centering, can't implement ExtrapolateAndZeroFace::apply()
// correctly, and can't partial-specialize yet, so... don't have a prototype
// for unspecified centering, so user gets a compile error if he tries to
// invoke it for a centering not yet implemented. Implement external functions
// which are specializations for the various centerings
// {Cell,Vert,CartesianCentering}; these are called from the general
// ExtrapolateAndZeroFace::apply() function body.
//----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////

template<class T, unsigned D, class M>
void ExtrapolateAndZeroFaceBCApply(ExtrapolateAndZeroFace<T,D,M,Cell>& ef,
			    Field<T,D,M,Cell>& A );
template<class T, unsigned D, class M>
void ExtrapolateAndZeroFaceBCApply(ExtrapolateAndZeroFace<T,D,M,Vert>& ef,
			    Field<T,D,M,Vert>& A );
template<class T, unsigned D, class M>
void ExtrapolateAndZeroFaceBCApply(ExtrapolateAndZeroFace<T,D,M,Edge>& ef,
			    Field<T,D,M,Edge>& A );
template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
void ExtrapolateAndZeroFaceBCApply(ExtrapolateAndZeroFace<T,D,M,
			    CartesianCentering<CE,D,NC> >& ef,
			    Field<T,D,M,CartesianCentering<CE,D,NC> >& A );

template<class T, unsigned D, class M, class C>
void ExtrapolateAndZeroFace<T,D,M,C>::apply( Field<T,D,M,C>& A )
{
  ExtrapolateAndZeroFaceBCApply(*this, A);
}


template<class T, unsigned D, class M, class C>
inline void
ExtrapolateAndZeroFaceBCApply2(const NDIndex<D> &dest,
  const NDIndex<D> &src, LField<T,D> &fill, LField<T,D> &from,
  const NDIndex<D> &from_alloc, ExtrapolateAndZeroFace<T,D,M,C> &ef)
{
  // If both the 'fill' and 'from' are compressed and applying the boundary
  // condition on the compressed value would result in no change to
  // 'fill' we don't need to uncompress.

  if (fill.IsCompressed() && from.IsCompressed())
    {
      // So far, so good. Let's check to see if the boundary condition
      // would result in filling the guard cells with a different value.

      T a = fill.getCompressedData(), aref = a;
      T b = from.getCompressedData();
      if (ef.getComponent() == BCondBase<T,D,M,C>::allComponents)
	{
	  OpExtrapolateAndZero<T> op(ef.getOffset(),ef.getSlope());
	  PETE_apply(op, a, b);
	}
      else
	{
	  OpExtrapolateAndZeroComponent<T>
	    op(ef.getOffset(),ef.getSlope(),ef.getComponent());
	  PETE_apply(op, a, b);
	}
      if (a == aref)
	{
	  // Yea! We're outta here.

	  return;
	}
    }

  // Well poop, we have no alternative but to uncompress the region
  // we're filling.

  fill.Uncompress();

  NDIndex<D> from_it = src.intersect(from_alloc);
  NDIndex<D> fill_it = dest.plugBase(from_it);

  // Build iterators for the copy...

  typedef typename LField<T,D>::iterator LFI;
  LFI lhs = fill.begin(fill_it);
  LFI rhs = from.begin(from_it);

  // And do the assignment.

  if (ef.getComponent() == BCondBase<T,D,M,C>::allComponents)
    {
      BrickExpression< D, LFI, LFI, OpExtrapolateAndZero<T> >
	(lhs, rhs,
         OpExtrapolateAndZero<T>(ef.getOffset(),ef.getSlope())).apply();
    }
  else
    {
      BrickExpression< D, LFI, LFI, OpExtrapolateAndZeroComponent<T> >
	(lhs, rhs,
         OpExtrapolateAndZeroComponent<T>
	 (ef.getOffset(),ef.getSlope(),ef.getComponent())).apply();
    }
}


template<class T, unsigned D, class M, class C>
inline void
ExtrapolateAndZeroFaceBCApply3(const NDIndex<D> &dest,
  LField<T,D> &fill, ExtrapolateAndZeroFace<T,D,M,C> &ef)
{
  // If the LField we're filling is compressed and setting the
  // cells/components to zero wouldn't make any difference, we don't
  // need to uncompress.

  if (fill.IsCompressed())
    {
      // So far, so good. Let's check to see if the boundary condition
      // would result in filling the guard cells with a different value.

      if (ef.getComponent() == BCondBase<T,D,M,C>::allComponents)
	{
	  if (fill.getCompressedData() == T(0))
	    return;
	}
      else
	{
          typedef typename AppTypeTraits<T>::Element_t T1;

	  //boo-boo for scalar types  T a = fill.getCompressedData();
	  //boo-boo for scalar types         if (a[ef.getComponent()] == T1(0))
	  //boo-boo for scalar types    return;

	  T a = fill.getCompressedData(), aref = a;
	  OpAssignComponent<T> op(ef.getComponent());
	  PETE_apply(op, a, T1(0));
	  if (a == aref)
	    return;
	}
    }

  // Well poop, we have no alternative but to uncompress the region
  // we're filling.

  fill.Uncompress();

  // Build iterator for the assignment...

  typedef typename LField<T,D>::iterator LFI;
  LFI lhs = fill.begin(dest);

  // And do the assignment.

  if (ef.getComponent() == BCondBase<T,D,M,C>::allComponents)
    {
      BrickExpression<D,LFI,PETE_Scalar<T>,OpAssign >
	(lhs,PETE_Scalar<T>(T(0)),OpAssign()).apply();
    }
  else
    {
      typedef typename AppTypeTraits<T>::Element_t T1;

      BrickExpression<D,LFI,PETE_Scalar<T1>,OpAssignComponent<T> >
	(lhs,PETE_Scalar<T1>(T1(0)),OpAssignComponent<T>
	 (ef.getComponent())).apply();
    }
}


//-----------------------------------------------------------------------------
// Specialization of ExtrapolateAndZeroFace::apply() for Cell centering.
// Rather, indirectly-called specialized global function
// ExtrapolateAndZeroFaceBCApply
//-----------------------------------------------------------------------------

template<class T, unsigned D, class M>
void ExtrapolateAndZeroFaceBCApply(ExtrapolateAndZeroFace<T,D,M,Cell>& ef,
			    Field<T,D,M,Cell>& A )
{



  // Find the slab that is the destination.
  // That is, in English, get an NDIndex spanning elements in the guard layers
  // on the face associated with the ExtrapaloteFace object:

  const NDIndex<D>& domain( A.getDomain() ); // Spans whole Field
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());

  // The direction (dimension of the Field) associated with the active face.
  // The numbering convention makes this division by two return the right
  // value, which will be between 0 and (D-1):

  unsigned d = ef.getFace()/2;
  int offset;

  // The following bitwise AND logical test returns true if ef.m_face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if ef.m_face is even (meaning the "low" or "left" face
  // in the numbering convention):

  if (ef.getFace() & 1)
    {
      // For "high" face, index in active direction goes from max index of
      // Field plus 1 to the same plus number of guard layers:
      // TJW: this used to say "leftGuard(d)", which I think was wrong:

      slab[d] = Index( domain[d].max() + 1, domain[d].max() + A.rightGuard(d));

      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:

      offset = 2*domain[d].max() + 1;
    }
  else
    {
      // For "low" face, index in active direction goes from min index of
      // Field minus the number of guard layers (usually a negative number)
      // to the same min index minus 1 (usually negative, and usually -1):

      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );

      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:

      offset = 2*domain[d].min() - 1;
    }

  // Loop over all the LField's in the Field A:

  typename Field<T,D,M,Cell>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      // Pointer to the data for the current LField (right????):

      LField<T,D> &fill = *(*fill_i).second;

      // NDIndex spanning all elements in the LField, including the guards:

      const NDIndex<D> &fill_alloc = fill.getAllocated();

      // If the previously-created boundary guard-layer NDIndex "slab"
      // contains any of the elements in this LField (they will be guard
      // elements if it does), assign the values into them here by applying
      // the boundary condition:

      if (slab.touches(fill_alloc))
        {
          // Find what it touches in this LField.

          NDIndex<D> dest = slab.intersect(fill_alloc);

          // For extrapolation boundary conditions, the boundary guard-layer
	  // elements are typically copied from interior values; the "src"
	  // NDIndex specifies the interior elements to be copied into the
	  // "dest" boundary guard-layer elements (possibly after some
	  // mathematical operations like multipplying by minus 1 later):

          NDIndex<D> src = dest;

	  // Now calculate the interior elements; the offset variable computed
	  // above makes this correct for "low" or "high" face cases:

          src[d] = offset - src[d];

	  // At this point, we need to see if 'src' is fully contained by
	  // by 'fill_alloc'. If it is, we have a lot less work to do.

	  if (fill_alloc.contains(src))
	    {
	      // Great! Our domain contains the elements we're filling from.

	      ExtrapolateAndZeroFaceBCApply2(dest, src, fill, fill,
	        fill_alloc, ef);
	    }
	  else
	    {
	      // Yuck! Our domain doesn't contain all of the src. We
	      // must loop over LFields to find the ones the touch the src.

	      typename Field<T,D,M,Cell>::iterator_if from_i;
	      for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
		{
		  // Cache a few things.

		  LField<T,D> &from = *(*from_i).second;
		  const NDIndex<D> &from_owned = from.getOwned();
		  const NDIndex<D> &from_alloc = from.getAllocated();

		  // If src touches this LField...

		  if (src.touches(from_owned))
		    ExtrapolateAndZeroFaceBCApply2(dest, src, fill, from,
		      from_alloc, ef);
		}
	    }
	}
    }
}


//-----------------------------------------------------------------------------
// Specialization of ExtrapolateAndZeroFace::apply() for Vert centering.
// Rather, indirectly-called specialized global function
// ExtrapolateAndZeroFaceBCApply
//-----------------------------------------------------------------------------

template<class T, unsigned D, class M>
void ExtrapolateAndZeroFaceBCApply(ExtrapolateAndZeroFace<T,D,M,Vert>& ef,
			    Field<T,D,M,Vert>& A )
{

  // Find the slab that is the destination.
  // That is, in English, get an NDIndex spanning elements in the guard layers
  // on the face associated with the ExtrapaloteFace object:

  const NDIndex<D>& domain(A.getDomain());
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  //boo-boo-2  NDIndex<D> phys = domain;
  NDIndex<D> phys = slab;

  // The direction (dimension of the Field) associated with the active face.
  // The numbering convention makes this division by two return the right
  // value, which will be between 0 and (D-1):

  unsigned d = ef.getFace()/2;
  int offset;

  // The following bitwise AND logical test returns true if ef.m_face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if ef.m_face is even (meaning the "low" or "left" face in
  // the numbering convention):

  if ( ef.getFace() & 1 )
    {
      // For "high" face, index in active direction goes from max index of
      // Field plus 1 to the same plus number of guard layers:
      // TJW: this used to say "leftGuard(d)", which I think was wrong:

      slab[d] = Index( domain[d].max() + 1, domain[d].max() + A.rightGuard(d));

      // Compute the layer of physical cells we're going to set.

      phys[d] = Index( domain[d].max(),  domain[d].max(), 1);

      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:
      // N.B.: the extra -1 here is what distinguishes this Vert-centered
      // implementation from the Cell-centered one:

      offset = 2*domain[d].max() + 1 - 1;
    }
  else
    {
      // For "low" face, index in active direction goes from min index of
      // Field minus the number of guard layers (usually a negative number)
      // to the same min index minus 1 (usually negative, and usually -1):

      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );

      // Compute the layer of physical cells we're going to set.

      phys[d] = Index( domain[d].min(),  domain[d].min(), 1);

      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:
      // N.B.: the extra +1 here is what distinguishes this Vert-centered
      // implementation from the Cell-centered one:

      offset = 2*domain[d].min() - 1 + 1;
    }

  // Loop over all the LField's in the Field A:

  typename Field<T,D,M,Vert>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      // Pointer to the data for the current LField (right????):

      LField<T,D> &fill = *(*fill_i).second;

      // Get the physical part of this LField and see if that touches
      // the physical cells we want to zero.

      //boo-boo-2      const NDIndex<D> &fill_owned = fill.getOwned();
      const NDIndex<D> &fill_alloc = fill.getAllocated();

      //boo-boo-2      if (phys.touches(fill_owned))
      if (phys.touches(fill_alloc))
	{
	  // Find out what we're touching.

	  //boo-boo-2	  NDIndex<D> dest = phys.intersect(fill_owned);
	  NDIndex<D> dest = phys.intersect(fill_alloc);

	  // Zero the cells.

	  ExtrapolateAndZeroFaceBCApply3(dest, fill, ef);
	}

      // NDIndex spanning all elements in the LField, including the guards:

      //boo-boo-2      const NDIndex<D> &fill_alloc = fill.getAllocated();

      // If the previously-created boundary guard-layer NDIndex "slab"
      // contains any of the elements in this LField (they will be guard
      // elements if it does), assign the values into them here by applying
      // the boundary condition:

      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.

          NDIndex<D> dest = slab.intersect( fill_alloc );

          // For exrapolation boundary conditions, the boundary guard-layer
	  // elements are typically copied from interior values; the "src"
	  // NDIndex specifies the interior elements to be copied into the
	  // "dest" boundary guard-layer elements (possibly after some
	  // mathematical operations like multipplying by minus 1 later):

          NDIndex<D> src = dest;

	  // Now calculate the interior elements; the offset variable computed
	  // above makes this correct for "low" or "high" face cases:

          src[d] = offset - src[d];

	  // At this point, we need to see if 'src' is fully contained by
	  // by 'fill_alloc'. If it is, we have a lot less work to do.

	  if (fill_alloc.contains(src))
	    {
	      // Great! Our domain contains the elements we're filling from.

	      ExtrapolateAndZeroFaceBCApply2(dest, src, fill, fill,
	        fill_alloc, ef);
	    }
	  else
	    {
	      // Yuck! Our domain doesn't contain all of the src. We
	      // must loop over LFields to find the ones the touch the src.

	      typename Field<T,D,M,Vert>::iterator_if from_i;
	      for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
		{
		  // Cache a few things.

		  LField<T,D> &from = *(*from_i).second;
		  const NDIndex<D> &from_owned = from.getOwned();
		  const NDIndex<D> &from_alloc = from.getAllocated();

		  // If src touches this LField...

		  if (src.touches(from_owned))
		    ExtrapolateAndZeroFaceBCApply2(dest, src, fill, from,
		      from_alloc, ef);
		}
	    }
	}
    }
}


//-----------------------------------------------------------------------------
// Specialization of ExtrapolateAndZeroFace::apply() for Edge centering.
// Rather, indirectly-called specialized global function
// ExtrapolateAndZeroFaceBCApply
//-----------------------------------------------------------------------------

template<class T, unsigned D, class M>
void ExtrapolateAndZeroFaceBCApply(ExtrapolateAndZeroFace<T,D,M,Edge>& ef,
                                   Field<T,D,M,Edge>& A )
{
  // Find the slab that is the destination.
  // That is, in English, get an NDIndex spanning elements in the guard layers
  // on the face associated with the ExtrapaloteFace object:

  const NDIndex<D>& domain(A.getDomain());
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  //boo-boo-2  NDIndex<D> phys = domain;
  NDIndex<D> phys = slab;

  // The direction (dimension of the Field) associated with the active face.
  // The numbering convention makes this division by two return the right
  // value, which will be between 0 and (D-1):

  unsigned d = ef.getFace()/2;
  int offset;

  // The following bitwise AND logical test returns true if ef.m_face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if ef.m_face is even (meaning the "low" or "left" face in
  // the numbering convention):

  if ( ef.getFace() & 1 )
    {
      // For "high" face, index in active direction goes from max index of
      // Field plus 1 to the same plus number of guard layers:
      // TJW: this used to say "leftGuard(d)", which I think was wrong:

      slab[d] = Index( domain[d].max() + 1, domain[d].max() + A.rightGuard(d));

      // Compute the layer of physical cells we're going to set.

      phys[d] = Index( domain[d].max(),  domain[d].max(), 1);

      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:
      // N.B.: the extra -1 here is what distinguishes this Edge-centered
      // implementation from the Cell-centered one:

      offset = 2*domain[d].max() + 1 - 1;
    }
  else
    {
      // For "low" face, index in active direction goes from min index of
      // Field minus the number of guard layers (usually a negative number)
      // to the same min index minus 1 (usually negative, and usually -1):

      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );

      // Compute the layer of physical cells we're going to set.

      phys[d] = Index( domain[d].min(),  domain[d].min(), 1);

      // Used in computing interior elements used in computing fill values for
      // boundary guard  elements; see below:
      // N.B.: the extra +1 here is what distinguishes this Edge-centered
      // implementation from the Cell-centered one:

      offset = 2*domain[d].min() - 1 + 1;
    }

  // Loop over all the LField's in the Field A:

  typename Field<T,D,M,Edge>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      // Pointer to the data for the current LField (right????):

      LField<T,D> &fill = *(*fill_i).second;

      // Get the physical part of this LField and see if that touches
      // the physical cells we want to zero.

      //boo-boo-2      const NDIndex<D> &fill_owned = fill.getOwned();
      const NDIndex<D> &fill_alloc = fill.getAllocated();

      //boo-boo-2      if (phys.touches(fill_owned))
      if (phys.touches(fill_alloc))
	{
	  // Find out what we're touching.

	  //boo-boo-2	  NDIndex<D> dest = phys.intersect(fill_owned);
	  NDIndex<D> dest = phys.intersect(fill_alloc);

	  // Zero the cells.

	  ExtrapolateAndZeroFaceBCApply3(dest, fill, ef);
	}

      // NDIndex spanning all elements in the LField, including the guards:

      //boo-boo-2      const NDIndex<D> &fill_alloc = fill.getAllocated();

      // If the previously-created boundary guard-layer NDIndex "slab"
      // contains any of the elements in this LField (they will be guard
      // elements if it does), assign the values into them here by applying
      // the boundary condition:

      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.

          NDIndex<D> dest = slab.intersect( fill_alloc );

          // For exrapolation boundary conditions, the boundary guard-layer
	  // elements are typically copied from interior values; the "src"
	  // NDIndex specifies the interior elements to be copied into the
	  // "dest" boundary guard-layer elements (possibly after some
	  // mathematical operations like multipplying by minus 1 later):

          NDIndex<D> src = dest;

	  // Now calculate the interior elements; the offset variable computed
	  // above makes this correct for "low" or "high" face cases:

          src[d] = offset - src[d];

	  // At this point, we need to see if 'src' is fully contained by
	  // by 'fill_alloc'. If it is, we have a lot less work to do.

	  if (fill_alloc.contains(src))
	    {
	      // Great! Our domain contains the elements we're filling from.

	      ExtrapolateAndZeroFaceBCApply2(dest, src, fill, fill,
	        fill_alloc, ef);
	    }
	  else
	    {
	      // Yuck! Our domain doesn't contain all of the src. We
	      // must loop over LFields to find the ones the touch the src.

	      typename Field<T,D,M,Edge>::iterator_if from_i;
	      for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
		{
		  // Cache a few things.

		  LField<T,D> &from = *(*from_i).second;
		  const NDIndex<D> &from_owned = from.getOwned();
		  const NDIndex<D> &from_alloc = from.getAllocated();

		  // If src touches this LField...

		  if (src.touches(from_owned))
		    ExtrapolateAndZeroFaceBCApply2(dest, src, fill, from,
		      from_alloc, ef);
		}
	    }
	}
    }
}

//-----------------------------------------------------------------------------
// Specialization of ExtrapolateAndZeroFace::apply() for CartesianCentering
// centering.  Rather,indirectly-called specialized global function
// ExtrapolateAndZeroFaceBCApply:
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
void ExtrapolateAndZeroFaceBCApply(ExtrapolateAndZeroFace<T,D,M,
			    CartesianCentering<CE,D,NC> >& ef,
			    Field<T,D,M,CartesianCentering<CE,D,NC> >& A )
{

  // Find the slab that is the destination.
  // That is, in English, get an NDIndex spanning elements in the guard layers
  // on the face associated with the ExtrapaloteFace object:

  const NDIndex<D>& domain( A.getDomain() ); // Spans whole Field
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  //boo-boo-2  NDIndex<D> phys = domain;
  NDIndex<D> phys = slab;

  // The direction (dimension of the Field) associated with the active face.
  // The numbering convention makes this division by two return the right
  // value, which will be between 0 and (D-1):

  unsigned d = ef.getFace()/2;
  int offset;
  bool setPhys = false;

  // The following bitwise AND logical test returns true if ef.m_face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if ef.m_face is even (meaning the "low" or "left" face in
  // the numbering convention):

  if ( ef.getFace() & 1 )
    {
      // offset is used in computing interior elements used in computing fill
      // values for boundary guard  elements; see below:
      // Do the right thing for CELL or VERT centering for this component (or
      // all components, if the PeriodicFace object so specifies):

      if (ef.getComponent() == BCondBase<T,D,M,CartesianCentering<CE,D,NC> >::
	  allComponents)
	{
	  // Make sure all components are really centered the same, as assumed:

	  CenteringEnum centering0 = CE[0 + d*NC]; // 1st component along dir d
	  for (unsigned int c=1; c<NC; c++)
	    {
	      // Compare other components with 1st
	      if (CE[c + d*NC] != centering0)
		ERRORMSG(
		  "ExtrapolateAndZeroFaceBCApply: BCond thinks all components"
			 << " have same centering along direction " << d
			 << ", but it isn't so." << endl);
	    }

	  // Now do the right thing for CELL or VERT centering of
	  // all components:

	  // For "high" face, index in active direction goes from max index of
	  // Field plus 1 to the same plus number of guard layers:

	  slab[d] = Index(domain[d].max() + 1,
			  domain[d].max() + A.rightGuard(d));

	  if (centering0 == CELL)
	    {
	      offset = 2*domain[d].max() + 1 ;    // CELL case
	    }
	  else
	    {
	      offset = 2*domain[d].max() + 1 - 1; // VERT case

	      // Compute the layer of physical cells we're going to set.

	      phys[d] = Index( domain[d].max(),  domain[d].max(), 1);
	      setPhys = true;
	    }
	}
      else
	{
	  // The BC applies only to one component, not all:
	  // Do the right thing for CELL or VERT centering of the component:
	  if (CE[ef.getComponent() + d*NC] == CELL)
	    {
	      // For "high" face, index in active direction goes from max index
	      // of cells in the Field plus 1 to the same plus number of guard
	      // layers:
	      int highcell = A.get_mesh().gridSizes[d] - 2;
	      slab[d] = Index(highcell + 1, highcell + A.rightGuard(d));

	      //	      offset = 2*domain[d].max() + 1 ;    // CELL case
	      offset = 2*highcell + 1 ;    // CELL case
	    }
	  else
	    {
	      // For "high" face, index in active direction goes from max index
	      // of verts in the Field plus 1 to the same plus number of guard
	      // layers:

	      int highvert = A.get_mesh().gridSizes[d] - 1;
	      slab[d] = Index(highvert + 1, highvert + A.rightGuard(d));

	      //	      offset = 2*domain[d].max() + 1 - 1; // VERT case

	      offset = 2*highvert + 1 - 1; // VERT case

	      // Compute the layer of physical cells we're going to set.

	      phys[d] = Index( highvert, highvert, 1 );
	      setPhys = true;
	    }
	}
    }
  else
    {
      // For "low" face, index in active direction goes from min index of
      // Field minus the number of guard layers (usually a negative number)
      // to the same min index minus 1 (usually negative, and usually -1):

      slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );

      // offset is used in computing interior elements used in computing fill
      // values for boundary guard  elements; see below:
      // Do the right thing for CELL or VERT centering for this component (or
      // all components, if the PeriodicFace object so specifies):

      if (ef.getComponent() == BCondBase<T,D,M,CartesianCentering<CE,D,NC> >::
	  allComponents)
	{
	  // Make sure all components are really centered the same, as assumed:

	  CenteringEnum centering0 = CE[0 + d*NC]; // 1st component along dir d
	  for (unsigned int c=1; c<NC; c++)
	    {
	      // Compare other components with 1st

	      if (CE[c + d*NC] != centering0)
		ERRORMSG(
                  "ExtrapolateAndZeroFaceBCApply: BCond thinks all components"
		     << " have same centering along direction " << d
		     << ", but it isn't so." << endl);
	    }

	  // Now do the right thing for CELL or VERT centering of all
	  // components:

	  if (centering0 == CELL)
	    {
	      offset = 2*domain[d].min() - 1;     // CELL case
	    }
	  else
	    {
	      offset = 2*domain[d].min() - 1 + 1; // VERT case

	      // Compute the layer of physical cells we're going to set.

	      phys[d] = Index(domain[d].min(),  domain[d].min(), 1);
	      setPhys = true;
	    }
	}
      else
	{
	  // The BC applies only to one component, not all:
	  // Do the right thing for CELL or VERT centering of the component:

	  if (CE[ef.getComponent() + d*NC] == CELL)
	    {
	      offset = 2*domain[d].min() - 1;     // CELL case
	    }
	  else
	    {
	      offset = 2*domain[d].min() - 1 + 1; // VERT case

	      // Compute the layer of physical cells we're going to set.

	      phys[d] = Index(domain[d].min(),  domain[d].min(), 1);
	      setPhys = true;
	    }
	}
    }

  // Loop over all the LField's in the Field A:

  typename Field<T,D,M,CartesianCentering<CE,D,NC> >::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      // Pointer to the data for the current LField (right????):

      LField<T,D> &fill = *(*fill_i).second;

      // Get the physical part of this LField and see if that touches
      // the physical cells we want to zero.

      const NDIndex<D> &fill_alloc = fill.getAllocated(); // moved here 1/27/99

      if (setPhys)
	{
	  //boo-boo-2	  const NDIndex<D> &fill_owned = fill.getOwned();

	  //boo-boo-2	  if (phys.touches(fill_owned))
	  if (phys.touches(fill_alloc))
	    {
	      // Find out what we're touching.

	      //boo-boo-2	      NDIndex<D> dest = phys.intersect(fill_owned);
	      NDIndex<D> dest = phys.intersect(fill_alloc);

	      // Zero the cells.

	      ExtrapolateAndZeroFaceBCApply3(dest, fill, ef);
	    }
	}

      // NDIndex spanning all elements in the LField, including the guards:

      //boo-boo-2      const NDIndex<D> &fill_alloc = fill.getAllocated();

      // If the previously-created boundary guard-layer NDIndex "slab"
      // contains any of the elements in this LField (they will be guard
      // elements if it does), assign the values into them here by applying
      // the boundary condition:

      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.

          NDIndex<D> dest = slab.intersect( fill_alloc );

          // For extrapolation boundary conditions, the boundary guard-layer
	  // elements are typically copied from interior values; the "src"
	  // NDIndex specifies the interior elements to be copied into the
	  // "dest" boundary guard-layer elements (possibly after some
	  // mathematical operations like multipplying by minus 1 later):

          NDIndex<D> src = dest;

	  // Now calculate the interior elements; the offset variable computed
	  // above makes this correct for "low" or "high" face cases:

          src[d] = offset - src[d];

	  // At this point, we need to see if 'src' is fully contained by
	  // by 'fill_alloc'. If it is, we have a lot less work to do.

	  if (fill_alloc.contains(src))
	    {
	      // Great! Our domain contains the elements we're filling from.

	      ExtrapolateAndZeroFaceBCApply2(dest, src, fill, fill,
	        fill_alloc, ef);
	    }
	  else
	    {
	      // Yuck! Our domain doesn't contain all of the src. We
	      // must loop over LFields to find the ones the touch the src.

	      typename Field<T,D,M,CartesianCentering<CE,D,NC> >::iterator_if
		from_i;
	      for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
		{
		  // Cache a few things.

		  LField<T,D> &from = *(*from_i).second;
		  const NDIndex<D> &from_owned = from.getOwned();
		  const NDIndex<D> &from_alloc = from.getAllocated();

		  // If src touches this LField...

		  if (src.touches(from_owned))
		    ExtrapolateAndZeroFaceBCApply2(dest, src, fill, from,
		      from_alloc, ef);
		}
	    }
	}
    }

}

// TJW added 12/16/1997 as per Tecolote Team's request... END
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

//////////////////////////////////////////////////////////////////////

// Applicative templates for FunctionFace:

// Standard, for applying to all components of elemental type:
template<class T>
struct OpBCFunctionEq
{
  OpBCFunctionEq(T (*func)(const T&)) : Func(func) {}
  T (*Func)(const T&);
};
template<class T>
//tjw3/12/1999inline void PETE_apply(const OpBCFunctionEq<T>& e, T& a, const T& b)
inline void PETE_apply(const OpBCFunctionEq<T>& e, T& a, T& b)
{
  a = e.Func(b);
}

// Special, for applying to single component of multicomponent elemental type:
// DOESN'T EXIST FOR FunctionFace; use ComponentFunctionFace instead.

//////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// For unspecified centering, can't implement FunctionFace::apply()
// correctly, and can't partial-specialize yet, so... don't have a prototype
// for unspecified centering, so user gets a compile error if he tries to
// invoke it for a centering not yet implemented. Implement external functions
// which are specializations for the various centerings
// {Cell,Vert,CartesianCentering}; these are called from the general
// FunctionFace::apply() function body.
//----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////

template<class T, unsigned D, class M>
void FunctionFaceBCApply(FunctionFace<T,D,M,Cell>& ff,
			 Field<T,D,M,Cell>& A );
template<class T, unsigned D, class M>
void FunctionFaceBCApply(FunctionFace<T,D,M,Vert>& ff,
			 Field<T,D,M,Vert>& A );
template<class T, unsigned D, class M>
void FunctionFaceBCApply(FunctionFace<T,D,M,Edge>& ff,
			 Field<T,D,M,Edge>& A );
template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
void FunctionFaceBCApply(FunctionFace<T,D,M,
			 CartesianCentering<CE,D,NC> >& ff,
			 Field<T,D,M,CartesianCentering<CE,D,NC> >& A );

template<class T, unsigned D, class M, class C>
void FunctionFace<T,D,M,C>::apply( Field<T,D,M,C>& A )
{


  FunctionFaceBCApply(*this, A);
}

//-----------------------------------------------------------------------------
// Specialization of FunctionFace::apply() for Cell centering.
// Rather, indirectly-called specialized global function FunctionFaceBCApply
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M>
void FunctionFaceBCApply(FunctionFace<T,D,M,Cell>& ff,
			 Field<T,D,M,Cell>& A )
{



  // NOTE: See the ExtrapolateFaceBCApply functions above for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = ff.getFace()/2;
  int offset;
  if ( ff.getFace() & 1 ) {
    // TJW: this used to say "leftGuard(d)", which I think was wrong:
    slab[d] = Index( domain[d].max() + 1,  domain[d].max() + A.rightGuard(d));
    offset = 2*domain[d].max() + 1;
  }
  else {
    slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
    offset = 2*domain[d].min() - 1;
  }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,Cell>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = offset - src[d];

          // Loop over the ones that src touches.
          typename Field<T,D,M,Cell>::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
		  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  BrickExpression<D,LFI,LFI,OpBCFunctionEq<T> >
		    (lhs,rhs,OpBCFunctionEq<T>(ff.Func)).apply();
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Specialization of FunctionFace::apply() for Vert centering.
// Rather, indirectly-called specialized global function FunctionFaceBCApply
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M>
void FunctionFaceBCApply(FunctionFace<T,D,M,Vert>& ff,
			 Field<T,D,M,Vert>& A )
{



  // NOTE: See the ExtrapolateFaceBCApply functions above for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = ff.getFace()/2;
  int offset;
  if ( ff.getFace() & 1 ) {
    // TJW: this used to say "leftGuard(d)", which I think was wrong:
    slab[d] = Index( domain[d].max() + 1,  domain[d].max() + A.rightGuard(d));
    // N.B.: the extra -1 here is what distinguishes this Vert-centered
    // implementation from the Cell-centered one:
    offset = 2*domain[d].max() + 1 - 1;
  }
  else {
    slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
    // N.B.: the extra +1 here is what distinguishes this Vert-centered
    // implementation from the Cell-centered one:
    offset = 2*domain[d].min() - 1 + 1;
  }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,Vert>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = offset - src[d];

          // Loop over the ones that src touches.
          typename Field<T,D,M,Vert>::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
		  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  BrickExpression<D,LFI,LFI,OpBCFunctionEq<T> >
		    (lhs,rhs,OpBCFunctionEq<T>(ff.Func)).apply();
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Specialization of FunctionFace::apply() for Edge centering.
// Rather, indirectly-called specialized global function FunctionFaceBCApply
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M>
void FunctionFaceBCApply(FunctionFace<T,D,M,Edge>& ff,
			 Field<T,D,M,Edge>& A )
{
  // NOTE: See the ExtrapolateFaceBCApply functions above for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = ff.getFace()/2;
  int offset;
  if ( ff.getFace() & 1 ) {
    // TJW: this used to say "leftGuard(d)", which I think was wrong:
    slab[d] = Index( domain[d].max() + 1,  domain[d].max() + A.rightGuard(d));
    // N.B.: the extra -1 here is what distinguishes this Edge-centered
    // implementation from the Cell-centered one:
    offset = 2*domain[d].max() + 1 - 1;
  }
  else {
    slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
    // N.B.: the extra +1 here is what distinguishes this Edge-centered
    // implementation from the Cell-centered one:
    offset = 2*domain[d].min() - 1 + 1;
  }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,Edge>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = offset - src[d];

          // Loop over the ones that src touches.
          typename Field<T,D,M,Edge>::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
		  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  BrickExpression<D,LFI,LFI,OpBCFunctionEq<T> >
		    (lhs,rhs,OpBCFunctionEq<T>(ff.Func)).apply();
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Specialization of FunctionFace::apply() for CartesianCentering centering.
// Rather, indirectly-called specialized global function FunctionFaceBCApply:
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
void FunctionFaceBCApply(FunctionFace<T,D,M,
			 CartesianCentering<CE,D,NC> >& ff,
			 Field<T,D,M,CartesianCentering<CE,D,NC> >& A )
{



  // NOTE: See the ExtrapolateFaceBCApply functions above for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = ff.getFace()/2;
  int offset;
  if ( ff.getFace() & 1 ) {
    // TJW: this used to say "leftGuard(d)", which I think was wrong:
    slab[d] = Index( domain[d].max() + 1,  domain[d].max() + A.rightGuard(d));
    // Do the right thing for CELL or VERT centering for all components:
    // Make sure all components are really centered the same, as assumed:
    CenteringEnum centering0 = CE[0 + d*NC]; // 1st component along dir d
    for (int c=1; c<NC; c++) { // Compare other components with 1st
      if (CE[c + d*NC] != centering0)
	ERRORMSG("FunctionFaceBCApply: BCond thinks all components have"
		 << " same centering along direction " << d
		 << ", but it isn't so." << endl);
    }
    // Now do the right thing for CELL or VERT centering of all components:
    if (centering0 == CELL) {
      offset = 2*domain[d].max() + 1;     // CELL case
    } else {
      offset = 2*domain[d].max() + 1 - 1; // VERT case
    }
  }
  else {
    slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
    // Do the right thing for CELL or VERT centering for all components:
    // Make sure all components are really centered the same, as assumed:
    CenteringEnum centering0 = CE[0 + d*NC]; // 1st component along dir d
    for (int c=1; c<NC; c++) { // Compare other components with 1st
      if (CE[c + d*NC] != centering0)
	ERRORMSG("FunctionFaceBCApply: BCond thinks all components have"
		 << " same centering along direction " << d
		 << ", but it isn't so." << endl);
    }
    // Now do the right thing for CELL or VERT centering of all components:
    if (centering0 == CELL) {
      offset = 2*domain[d].min() - 1;     // CELL case
    } else {
      offset = 2*domain[d].min() - 1 + 1; // VERT case
    }
  }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,CartesianCentering<CE,D,NC> >::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = offset - src[d];

          // Loop over the ones that src touches.
          typename Field<T,D,M,CartesianCentering<CE,D,NC> >::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
		  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  BrickExpression<D,LFI,LFI,OpBCFunctionEq<T> >
		    (lhs,rhs,OpBCFunctionEq<T>(ff.Func)).apply();
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////

// Applicative templates for ComponentFunctionFace:

// Standard, for applying to all components of elemental type:
// DOESN'T EXIST FOR ComponentFunctionFace; use FunctionFace instead.

// Special, for applying to single component of multicomponent elemental type:
template<class T>
struct OpBCFunctionEqComponent
{
  OpBCFunctionEqComponent(typename ApplyToComponentType<T>::type (*func)
			  ( typename ApplyToComponentType<T>::type ),
                          int c) :
    Func(func), Component(c) {}
  typename ApplyToComponentType<T>::type
    (*Func)( typename ApplyToComponentType<T>::type );
  int Component;
};
template<class T>
inline void PETE_apply(const OpBCFunctionEqComponent<T>& e, T& a, const T& b)
{ a[e.Component] = e.Func(b[e.Component]); }

// Following specializations are necessary because of the runtime branches in
// code which unfortunately force instantiation of OpBCFunctionEqComponent
// instances for non-multicomponent types like {char,double,...}.
// Note: if user uses non-multicomponent (no operator[]) types of his own,
// he'll get a compile error. See comments regarding similar specializations
// for OpExtrapolateComponent for a more details.

COMPONENT_APPLY_BUILTIN(OpBCFunctionEqComponent,char)
COMPONENT_APPLY_BUILTIN(OpBCFunctionEqComponent,bool)
COMPONENT_APPLY_BUILTIN(OpBCFunctionEqComponent,int)
COMPONENT_APPLY_BUILTIN(OpBCFunctionEqComponent,unsigned)
COMPONENT_APPLY_BUILTIN(OpBCFunctionEqComponent,short)
COMPONENT_APPLY_BUILTIN(OpBCFunctionEqComponent,long)
COMPONENT_APPLY_BUILTIN(OpBCFunctionEqComponent,float)
COMPONENT_APPLY_BUILTIN(OpBCFunctionEqComponent,double)
COMPONENT_APPLY_BUILTIN(OpBCFunctionEqComponent,std::complex<double>)


//////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// For unspecified centering, can't implement ComponentFunctionFace::apply()
// correctly, and can't partial-specialize yet, so... don't have a prototype
// for unspecified centering, so user gets a compile error if he tries to
// invoke it for a centering not yet implemented. Implement external functions
// which are specializations for the various centerings
// {Cell,Vert,CartesianCentering}; these are called from the general
// ComponentFunctionFace::apply() function body.
//----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////

template<class T, unsigned D, class M>
void ComponentFunctionFaceBCApply(ComponentFunctionFace<T,D,M,Cell>& ff,
				  Field<T,D,M,Cell>& A );
template<class T, unsigned D, class M>
void ComponentFunctionFaceBCApply(ComponentFunctionFace<T,D,M,Vert>& ff,
				  Field<T,D,M,Vert>& A );
template<class T, unsigned D, class M>
void ComponentFunctionFaceBCApply(ComponentFunctionFace<T,D,M,Edge>& ff,
				  Field<T,D,M,Edge>& A );
template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
void ComponentFunctionFaceBCApply(ComponentFunctionFace<T,D,M,
				  CartesianCentering<CE,D,NC> >& ff,
				  Field<T,D,M,
				  CartesianCentering<CE,D,NC> >& A );

template<class T, unsigned D, class M, class C>
void ComponentFunctionFace<T,D,M,C>::apply( Field<T,D,M,C>& A )
{
  ComponentFunctionFaceBCApply(*this, A);
}

//-----------------------------------------------------------------------------
//Specialization of ComponentFunctionFace::apply() for Cell centering.
//Rather, indirectly-called specialized global function
//ComponentFunctionFaceBCApply
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M>
void ComponentFunctionFaceBCApply(ComponentFunctionFace<T,D,M,Cell>& ff,
				  Field<T,D,M,Cell>& A )
{



  // NOTE: See the ExtrapolateFaceBCApply functions above for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = ff.getFace()/2;
  int offset;
  if ( ff.getFace() & 1 ) {
    // TJW: this used to say "leftGuard(d)", which I think was wrong:
    slab[d] = Index( domain[d].max() + 1,  domain[d].max() + A.rightGuard(d));
    offset = 2*domain[d].max() + 1;
  }
  else {
    slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
    offset = 2*domain[d].min() - 1;
  }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,Cell>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = offset - src[d];

          // Loop over the ones that src touches.
          typename Field<T,D,M,Cell>::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
		  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  BrickExpression<D,LFI,LFI,OpBCFunctionEqComponent<T> >
		    (lhs,rhs,OpBCFunctionEqComponent<T>
		     (ff.Func,ff.getComponent())).apply();
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
//Specialization of ComponentFunctionFace::apply() for Vert centering.
//Rather, indirectly-called specialized global function
//ComponentFunctionFaceBCApply
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M>
void ComponentFunctionFaceBCApply(ComponentFunctionFace<T,D,M,Vert>& ff,
			 Field<T,D,M,Vert>& A )
{



  // NOTE: See the ExtrapolateFaceBCApply functions above for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = ff.getFace()/2;
  int offset;
  if ( ff.getFace() & 1 ) {
    // TJW: this used to say "leftGuard(d)", which I think was wrong:
    slab[d] = Index( domain[d].max() + 1,  domain[d].max() + A.rightGuard(d));
    // N.B.: the extra -1 here is what distinguishes this Vert-centered
    // implementation from the Cell-centered one:
    offset = 2*domain[d].max() + 1 - 1;
  }
  else {
    slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
    // N.B.: the extra +1 here is what distinguishes this Vert-centered
    // implementation from the Cell-centered one:
    offset = 2*domain[d].min() - 1 + 1;
  }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,Vert>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = offset - src[d];

          // Loop over the ones that src touches.
          typename Field<T,D,M,Vert>::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
		  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  BrickExpression<D,LFI,LFI,OpBCFunctionEqComponent<T> >
		    (lhs,rhs,OpBCFunctionEqComponent<T>
		     (ff.Func,ff.getComponent())).apply();
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
//Specialization of ComponentFunctionFace::apply() for Edge centering.
//Rather, indirectly-called specialized global function
//ComponentFunctionFaceBCApply
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M>
void ComponentFunctionFaceBCApply(ComponentFunctionFace<T,D,M,Edge>& ff,
			 Field<T,D,M,Edge>& A )
{
  // NOTE: See the ExtrapolateFaceBCApply functions above for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = ff.getFace()/2;
  int offset;
  if ( ff.getFace() & 1 ) {
    // TJW: this used to say "leftGuard(d)", which I think was wrong:
    slab[d] = Index( domain[d].max() + 1,  domain[d].max() + A.rightGuard(d));
    // N.B.: the extra -1 here is what distinguishes this Edge-centered
    // implementation from the Cell-centered one:
    offset = 2*domain[d].max() + 1 - 1;
  }
  else {
    slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
    // N.B.: the extra +1 here is what distinguishes this Edge-centered
    // implementation from the Cell-centered one:
    offset = 2*domain[d].min() - 1 + 1;
  }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,Edge>::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = offset - src[d];

          // Loop over the ones that src touches.
          typename Field<T,D,M,Edge>::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
		  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  BrickExpression<D,LFI,LFI,OpBCFunctionEqComponent<T> >
		    (lhs,rhs,OpBCFunctionEqComponent<T>
		     (ff.Func,ff.getComponent())).apply();
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
//Specialization of ComponentFunctionFace::apply() for
//CartesianCentering centering.  Rather, indirectly-called specialized
//global function ComponentFunctionFaceBCApply:
//-----------------------------------------------------------------------------
template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
void ComponentFunctionFaceBCApply(ComponentFunctionFace<T,D,M,
			 CartesianCentering<CE,D,NC> >& ff,
			 Field<T,D,M,CartesianCentering<CE,D,NC> >& A )
{



  // NOTE: See the ExtrapolateFaceBCApply functions above for more
  // comprehensible comments --TJW

  // Find the slab that is the destination.
  const NDIndex<D>& domain( A.getDomain() );
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());
  unsigned d = ff.getFace()/2;
  int offset;
  if ( ff.getFace() & 1 ) {
    // TJW: this used to say "leftGuard(d)", which I think was wrong:
    slab[d] = Index( domain[d].max() + 1,  domain[d].max() + A.rightGuard(d));
    // Do the right thing for CELL or VERT centering for this component. (The
    // ComponentFunctionFace BC always applies only to one component, not all):
    if (CE[ff.getComponent() + d*NC] == CELL) {   // ada: changed ef to ff
        ERRORMSG("check src code, had to change ef (not known) to ff ??? " << endl);
        offset = 2*domain[d].max() + 1;     // CELL case
    } else {
        offset = 2*domain[d].max() + 1 - 1; // VERT case
    }
  }
  else {
    slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );
    // Do the right thing for CELL or VERT centering for this component. (The
    // ComponentFunctionFace BC always applies only to one component, not all):
    if (CE[ff.getComponent() + d*NC] == CELL) {
      offset = 2*domain[d].min() - 1;     // CELL case
    } else {
      offset = 2*domain[d].min() - 1 + 1; // VERT case
    }
  }

  // Loop over the ones the slab touches.
  typename Field<T,D,M,CartesianCentering<CE,D,NC> >::iterator_if fill_i;
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i)
    {
      // Cache some things we will use often below.
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      if ( slab.touches( fill_alloc ) )
        {
          // Find what it touches in this LField.
          NDIndex<D> dest = slab.intersect( fill_alloc );

          // Find where that comes from.
          NDIndex<D> src = dest;
          src[d] = offset - src[d];

          // Loop over the ones that src touches.
          typename Field<T,D,M,CartesianCentering<CE,D,NC> >::iterator_if from_i;
          for (from_i=A.begin_if(); from_i!=A.end_if(); ++from_i)
            {
              // Cache a few things.
              LField<T,D> &from = *(*from_i).second;
              const NDIndex<D> &from_owned = from.getOwned();
              const NDIndex<D> &from_alloc = from.getAllocated();
              // If src touches this LField...
              if ( src.touches( from_owned ) )
                {
		  // bfh: Worry about compression ...
		  // If 'fill' is a compressed LField, then writing data into
		  // it via the expression will actually write the value for
		  // all elements of the LField.  We can do the following:
		  //   a) figure out if the 'fill' elements are all the same
		  //      value, if 'from' elements are the same value, if
		  //      the 'fill' elements are the same as the elements
		  //      throughout that LField, and then just do an
		  //      assigment for a single element
		  //   b) just uncompress the 'fill' LField, to make sure we
		  //      do the right thing.
		  // I vote for b).
		  fill.Uncompress();

                  NDIndex<D> from_it = src.intersect( from_alloc );
                  NDIndex<D> fill_it = dest.plugBase( from_it );
                  // Build iterators for the copy...
		  typedef typename LField<T,D>::iterator LFI;
                  LFI lhs = fill.begin(fill_it);
                  LFI rhs = from.begin(from_it);
                  // And do the assignment.
		  BrickExpression<D,LFI,LFI,OpBCFunctionEqComponent<T> >
		    (lhs,rhs,OpBCFunctionEqComponent<T>
		     (ff.Func,ff.getComponent())).apply();
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////

//
// EurekaFace::apply().
// Apply a Eureka condition on a face.
//
// The general idea is to set the guard cells plus one layer
// of cells to zero in all cases except one:
// When there are mixed centerings, the cell coordinates just
// have their guard cells set to zero.
//

//------------------------------------------------------------
// The actual member function EurekaFace::apply
// This just uses the functions above to find the slab
// we are going to fill, then it fills it.
//------------------------------------------------------------

template<class T, unsigned int D, class M, class C>
void EurekaFace<T,D,M,C>::apply(Field<T,D,M,C>& field)
{
  // Calculate the domain we're going to fill
  // using the domain of the field.
  NDIndex<D> slab = calcEurekaSlabToFill(field,BCondBase<T,D,M,C>::m_face,this->getComponent());

  // Fill that slab.
  fillSlabWithZero(field,slab,this->getComponent());
}

//////////////////////////////////////////////////////////////////////

//
// Tag class for the Eureka assignment.
// This has two functions:
//
// A static member function for getting a component if
// there are subcomponents.
//
// Be a tag class for the BrickExpression for an assignment.
//

//
// First we have the general template class.
// This ignores the component argument, and will be used
// for any classes except Vektor, Tenzor and SymTenzor.
//

template<class T>
struct EurekaAssign
{
  static const T& get(const T& x, int) {
    ERRORMSG("Eureka assign to a component of class without an op[]!"<<endl);
    return x;
  }
  static T& get(T& x, int) {
    ERRORMSG("Eureka assign to a component of class without an op[]!"<<endl);
    return x;
  }
  int m_component;
  EurekaAssign(int c) : m_component(c) {}
};

//------------------------------------------------------------
// Given a Field and a domain, fill the domain with zeros.
// Input:
//   field:       The field we'll be assigning to.
//   fillDomain:  The domain we'll be assigning to.
//   component:   What component of T we'll be filling.
//------------------------------------------------------------

template<class T, unsigned int D, class M, class C>
static void
fillSlabWithZero(Field<T,D,M,C>& field,
		 const NDIndex<D>& fillDomain,
		 int component)
{
  //
  // Loop over all the vnodes, filling each one in turn.
  //
  typename Field<T,D,M,C>::iterator_if lp = field.begin_if();
  typename Field<T,D,M,C>::iterator_if done = field.end_if();
  for (; lp != done ; ++lp )
    {
      // Get a reference to the current LField.
      LField<T,D>& lf = *(*lp).second;

      // Get its domain, including guard cells.
      NDIndex<D> localDomain = lf.getAllocated();

      // If localDomain intersects fillDomain, we have work to do.
      if ( fillDomain.touches(localDomain) )
        {
          // If lf is compressed, we may not have to do any work.
          if ( lf.IsCompressed() )
            {
              // Check and see if we're dealing with all components
              if ( component == BCondBase<T,D,M,C>::allComponents )
                {
                  // If it is compressed to zero, we really don't have to work
                  if ( *lf.begin() == 0 )
                    continue;
                }
              // We are dealing with just one component
              else
                {
                  // Check to see if the component is zero.
                  // Check through a class so that this statement
                  // isn't a compile error if T is something like int.
                  if ( EurekaAssign<T>::get(*lf.begin(),component)==0 )
                    continue;
                }
              // Not compressed to zero.
              // Have to uncompress.
              lf.Uncompress();
            }

          // Get the actual intersection.
          NDIndex<D> intersectDomain = fillDomain.intersect(localDomain);

          // Get an iterator that will loop over that domain.
          typename LField<T,D>::iterator data = lf.begin(intersectDomain);


          // Build a BrickExpression that will fill it with zeros.
          // First some typedefs for the constituents of the expression.
          typedef typename LField<T,D>::iterator Lhs_t;
          typedef PETE_Scalar<T> Rhs_t;

          // If we are assigning all components, use regular assignment.
          if ( component == BCondBase<T,D,M,C>::allComponents )
            {
              // The type of the expression.
              typedef BrickExpression<D,Lhs_t,Rhs_t,OpAssign> Expr_t;

              // Build the expression and evaluate it.
              Expr_t(data,Rhs_t(0)).apply();
            }
          // We are assigning just one component. Use EurekaAssign.
          else
            {
              // The type of the expression.
	      typedef BrickExpression<D,Lhs_t,Rhs_t,EurekaAssign<T> > Expr_t;

	      // Sanity check.
	      PAssert_GE(component, 0);

              // Build the expression and evaluate it. tjw:mwerks dies here:
	      Expr_t(data,Rhs_t(0),component).apply();
	      //try:	      typedef typename AppTypeTraits<T>::Element_t T1;
	      //try:              Expr_t(data,PETE_Scalar<T1>(T1(0)),component).apply();
            }
        }
    }
}

//
// Specializations of EurekaAssign for Vektor, Tenzor, SymTenzor.
//

template<class T, unsigned int D>
struct EurekaAssign< Vektor<T,D> >
{
  static T& get( Vektor<T,D>& x, int c ) { return x[c]; }
  static T  get( const Vektor<T,D>& x, int c ) { return x[c]; }
  EurekaAssign(int c) : m_component(c) {}
  int m_component;
};

template<class T, unsigned int D>
struct EurekaAssign< Tenzor<T,D> >
{
  static T& get( Tenzor<T,D>& x, int c ) { return x[c]; }
  static T  get( const Tenzor<T,D>& x, int c ) { return x[c]; }
  EurekaAssign(int c) : m_component(c) {}
  int m_component;
};

template<class T, unsigned int D>
struct EurekaAssign< AntiSymTenzor<T,D> >
{
  static T& get( AntiSymTenzor<T,D>& x, int c ) { return x[c]; }
  static T  get( const AntiSymTenzor<T,D>& x, int c ) { return x[c]; }
  EurekaAssign(int c) : m_component(c) {}
  int m_component;
};

template<class T, unsigned int D>
struct EurekaAssign< SymTenzor<T,D> >
{
  static T& get( SymTenzor<T,D>& x, int c ) { return x[c]; }
  static T  get( const SymTenzor<T,D>& x, int c ) { return x[c]; }
  EurekaAssign(int c) : m_component(c) {}
  int m_component;
};

//
// A version of PETE_apply for EurekaAssign.
// Assign a component of the right hand side to a component of the left.
//

template<class T>
inline void PETE_apply(const EurekaAssign<T>& e, T& a, const T& b)
{
  EurekaAssign<T>::get(a,e.m_component) =
    EurekaAssign<T>::get(b,e.m_component);
}

//////////////////////////////////////////////////////////////////////

//
// calcEurekaDomain
// Given a domain, a face number, and a guard cell spec,
// find the low and high bounds for the domain we will be filling.
// Return the two integers through references.
//
template<unsigned int D>
inline NDIndex<D>
calcEurekaDomain(const NDIndex<D>& realDomain,
                 int face,
                 const GuardCellSizes<D>& gc)
{
  NDIndex<D> slab = AddGuardCells(realDomain,gc);

  // Find the dimension the slab will be perpendicular to.
  int dim = face/2;

  // The upper and lower bounds of the domain.
  int low,high;

  // If the face is odd, then it is the "high" end of the dimension.
  if ( face&1 )
    {
      // Find the bounds of the domain.
      high = slab[dim].max();
      low  = high - gc.right(dim);
    }
  // If the face is even, then it is the "low" end of the dimension.
  else
    {
      // Find the bounds of the domain.
      low  = slab[dim].min();
      high = low + gc.left(dim);
    }

  // Sanity checks.
  PAssert_LE( low, high );
  PAssert_LE( high, slab[dim].max() );
  PAssert_GE( low, slab[dim].min() );

  // Build the domain.
  slab[dim] = Index(low,high);
  return slab;
}

//////////////////////////////////////////////////////////////////////

//
// Find the appropriate slab to fill for a Cell centered field
// for the Eureka boundary condition.
// It's thickness is the number of guard cells plus one.
//

template<class T, unsigned int D, class M>
static NDIndex<D>
calcEurekaSlabToFill(const Field<T,D,M,Cell>& field, int face,int)
{
  return calcEurekaDomain(field.getDomain(),face,field.getGC());
}

template<class T, unsigned int D, class M>
static NDIndex<D>
calcEurekaSlabToFill(const Field<T,D,M,Vert>& field, int face,int)
{
  return calcEurekaDomain(field.getDomain(),face,field.getGC());
}

template<class T, unsigned int D, class M>
static NDIndex<D>
calcEurekaSlabToFill(const Field<T,D,M,Edge>& field, int face,int)
{
  return calcEurekaDomain(field.getDomain(),face,field.getGC());
}

//////////////////////////////////////////////////////////////////////

//
// Find the appropriate slab to fill for a mixed centering
// for the Eureka boundary condition.
// It's thickness is:
// If we're filling allComponents, the number guard cells plus 1.
// If we're filling a single component,
//   If that component if vert, the number of guard cells plus 1
//   If that component is cell, the number of guard cells.
//

template<class T, unsigned D, class M, CenteringEnum* CE, unsigned NC>
static NDIndex<D>
calcEurekaSlabToFill(const Field<T,D,M,CartesianCentering<CE,D,NC> >& field,
                     int face,
                     int component)
{
  // Shorthand for the centering type.
  typedef CartesianCentering<CE,D,NC> C;

  // Find the dimension perpendicular to the slab.
  int d = face/2;

  // The domain we'll be calculating.
  NDIndex<D> slab;

  // First check and see if we are fill all the components.
  if ( component == BCondBase<T,D,M,C>::allComponents )
    {
      // Deal with this like a pure VERT or CELL case.
      slab = calcEurekaDomain(field.getDomain(),face,field.getGC());
    }
  // Only do one component.
  else
    {
      // Our initial approximation to the slab to fill is the whole domain.
      // We'll reset the index for dimension d to make it a slab.
      slab = AddGuardCells(field.getDomain(),field.getGC());

      // If face is odd, this is the "high" face.
      if ( face&1 )
        {
          // Find the upper and lower bounds of the domain.
          int low = field.getDomain()[d].min() + field.get_mesh().gridSizes[d] - 1;
          int high = low + field.rightGuard(d);

          // For cell centered, we do one fewer layer of guard cells.
          if (CE[component + d*NC] == CELL)
	    low++;

	  // Sanity checks.
	  PAssert_LE( low, high );
	  PAssert_LE( high, slab[d].max() );
	  PAssert_GE( low, slab[d].min() );

          // Record this part of the slab.
          slab[d] = Index(low,high);
        }
      // If face is even, this is the "low" face.
      else
        {
          // Get the lower and upper bounds of the domain.
          int low = slab[d].min();
          int high = low + field.leftGuard(d) ;

          // For cell centered, we do one fewer layer of guard cells.
          if (CE[component + d*NC] == CELL)
            high--;

	  // Sanity checks.
	  PAssert_LE( low, high );
	  PAssert_LE( high, slab[d].max() );
	  PAssert_GE( low, slab[d].min() );

          // Record this part of the slab.
          slab[d] = Index(low,high);
        }
    }

  // We've found the domain, so return it.
  return slab;
}

//////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// For unspecified centering, can't implement LinearExtrapolateFace::apply()
// correctly, and can't partial-specialize yet, so... don't have a prototype
// for unspecified centering, so user gets a compile error if he tries to
// invoke it for a centering not yet implemented. Implement external functions
// which are specializations for the various centerings
// {Cell,Vert,CartesianCentering}; these are called from the general
// LinearExtrapolateFace::apply() function body.
//
// TJW: Actually, for LinearExtrapolate, don't need to specialize on
// centering. Probably don't need this indirection here, but leave it in for
// now.
//----------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////

template<class T, unsigned D, class M, class C>
void LinearExtrapolateFaceBCApply(LinearExtrapolateFace<T,D,M,C>& ef,
				  Field<T,D,M,C>& A );

template<class T, unsigned D, class M, class C>
void LinearExtrapolateFace<T,D,M,C>::apply( Field<T,D,M,C>& A )
{
  LinearExtrapolateFaceBCApply(*this, A);
}


template<class T, unsigned D, class M, class C>
inline void
LinearExtrapolateFaceBCApply2(const NDIndex<D> &dest,
				   const NDIndex<D> &src1,
				   const NDIndex<D> &src2,
				   LField<T,D> &fill,
                                   LinearExtrapolateFace<T,D,M,C> &/*ef*/,
				   int slopeMultipplier)
{
  // If 'fill' is compressed and applying the boundary condition on the
  // compressed value would result in no change to 'fill' we don't need to
  // uncompress.  For this particular type of BC (linear extrapolation), this
  // result would *never* happen, so we already know we're done:

  if (fill.IsCompressed()) { return; } // Yea! We're outta here.

  // Build iterators for the copy:
  typedef typename LField<T,D>::iterator LFI;
  LFI lhs  = fill.begin(dest);
  LFI rhs1 = fill.begin(src1);
  LFI rhs2 = fill.begin(src2);
  LFI endi = fill.end(); // Used for testing end of *any* sub-range iteration

  // Couldn't figure out how to use BrickExpression here. Just iterate through
  // all the elements in all 3 LField iterators (which are BrickIterators) and
  // do the calculation one element at a time:
  for ( ; lhs != endi && rhs1 != endi && rhs2 != endi;
	++lhs, ++rhs1, ++rhs2) {
    *lhs = (*rhs2 - *rhs1)*slopeMultipplier + *rhs1;
  }

}


// ----------------------------------------------------------------------------
// This type of boundary condition (linear extrapolation) does very much the
// same thing for any centering; Doesn't seem to be a need for specializations
// based on centering.
// ----------------------------------------------------------------------------

template<class T, unsigned D, class M, class C>
void LinearExtrapolateFaceBCApply(LinearExtrapolateFace<T,D,M,C>& ef,
				  Field<T,D,M,C>& A )
{



  // Find the slab that is the destination.
  // That is, in English, get an NDIndex spanning elements in the guard layers
  // on the face associated with the LinearExtrapaloteFace object:

  const NDIndex<D>& domain( A.getDomain() ); // Spans whole Field
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());

  // The direction (dimension of the Field) associated with the active face.
  // The numbering convention makes this division by two return the right
  // value, which will be between 0 and (D-1):

  unsigned d = ef.getFace()/2;

  // Must loop explicitly over the number of guard layers:
  int nGuardLayers;

  // The following bitwise AND logical test returns true if ef.m_face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if ef.m_face is even (meaning the "low" or "left" face in
  // the numbering convention):

  if (ef.getFace() & 1) {

    // For "high" face, index in active direction goes from max index of
    // Field plus 1 to the same plus number of guard layers:
    nGuardLayers = A.rightGuard(d);

  } else {

    // For "low" face, index in active direction goes from min index of
    // Field minus the number of guard layers (usually a negative number)
    // to the same min index minus 1 (usually negative, and usually -1):
    nGuardLayers = A.leftGuard(d);

  }

  // Loop over the number of guard layers, treating each layer as a separate
  // slab (contrast this with all other BC types, where all layers are a single
  // slab):
  for (int guardLayer = 1; guardLayer <= nGuardLayers; guardLayer++) {

    // For linear extrapolation, the multipplier increases with more distant
    // guard layers:
    int slopeMultipplier = -1*guardLayer;

    if (ef.getFace() & 1) {
      slab[d] = Index(domain[d].max() + guardLayer,
		      domain[d].max() + guardLayer);
    } else {
      slab[d] = Index(domain[d].min() - guardLayer,
		      domain[d].min() - guardLayer);
    }

    // Loop over all the LField's in the Field A:

    typename Field<T,D,M,Cell>::iterator_if fill_i;
    for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i) {

      // Cache some things we will use often below.

      // Pointer to the data for the current LField:
      LField<T,D> &fill = *(*fill_i).second;

      // NDIndex spanning all elements in the LField, including the guards:
      const NDIndex<D> &fill_alloc = fill.getAllocated();

      // If the previously-created boundary guard-layer NDIndex "slab"
      // contains any of the elements in this LField (they will be guard
      // elements if it does), assign the values into them here by applying
      // the boundary condition:

      if (slab.touches(fill_alloc)) {

	// Find what it touches in this LField.
	NDIndex<D> dest = slab.intersect(fill_alloc);

	// For linear extrapolation boundary conditions, the boundary
	// guard-layer elements are filled based on a slope and intercept
	// derived from two layers of interior values; the src1 and src2
	// NDIndexes specify these two interior layers, which are operated on
	// by mathematical operations whose results are put into dest.  The
	// ordering of what is defined as src1 and src2 is set differently for
	// hi and lo faces, to make the sign for extrapolation work out right:
	NDIndex<D> src1 = dest;
	NDIndex<D> src2 = dest;
	if (ef.getFace() & 1) {
	  src2[d] = Index(domain[d].max() - 1, domain[d].max() - 1, 1);
	  src1[d] = Index(domain[d].max(), domain[d].max(), 1);
	} else {
	  src1[d] = Index(0,0,1); // Note: hardwired to 0-base, stride-1; could
	  src2[d] = Index(1,1,1); //  generalize with domain.min(), etc.
	}

	// Assume that src1 and src2 are contained withi nthe fill_alloc LField
	// domain; I think this is always true if the vnodes are always at
	// least one guard-layer-width wide in number of physical elements:

	LinearExtrapolateFaceBCApply2(dest, src1, src2, fill, ef,
				      slopeMultipplier);

      }
    }
  }
}


//////////////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------------
// For unspecified centering, can't implement
// ComponentLinearExtrapolateFace::apply() correctly, and can't
// partial-specialize yet, so... don't have a prototype for unspecified
// centering, so user gets a compile error if he tries to invoke it for a
// centering not yet implemented. Implement external functions which are
// specializations for the various centerings {Cell,Vert,CartesianCentering};
// these are called from the general ComponentLinearExtrapolateFace::apply()
// function body.
//
// TJW: Actually, for ComponentLinearExtrapolate, don't need to specialize on
// centering. Probably don't need this indirection here, but leave it in for
// now.
// ---------------------------------------------------------------------------

template<class T, unsigned D, class M, class C>
void ComponentLinearExtrapolateFaceBCApply(ComponentLinearExtrapolateFace
					   <T,D,M,C>& ef,
					   Field<T,D,M,C>& A );

template<class T, unsigned D, class M, class C>
void ComponentLinearExtrapolateFace<T,D,M,C>::apply( Field<T,D,M,C>& A )
{
  ComponentLinearExtrapolateFaceBCApply(*this, A);
}


template<class T, unsigned D, class M, class C>
inline void
ComponentLinearExtrapolateFaceBCApply2(const NDIndex<D> &dest,
					    const NDIndex<D> &src1,
					    const NDIndex<D> &src2,
					    LField<T,D> &fill,
					    ComponentLinearExtrapolateFace
					    <T,D,M,C> &ef,
					    int slopeMultipplier)
{
  // If 'fill' is compressed and applying the boundary condition on the
  // compressed value would result in no change to 'fill' we don't need to
  // uncompress.  For this particular type of BC (linear extrapolation), this
  // result would *never* happen, so we already know we're done:

  if (fill.IsCompressed()) { return; } // Yea! We're outta here.

  // Build iterators for the copy:
  typedef typename LField<T,D>::iterator LFI;
  LFI lhs  = fill.begin(dest);
  LFI rhs1 = fill.begin(src1);
  LFI rhs2 = fill.begin(src2);
  LFI endi = fill.end(); // Used for testing end of *any* sub-range iteration

  // Couldn't figure out how to use BrickExpression here. Just iterate through
  // all the elements in all 3 LField iterators (which are BrickIterators) and
  // do the calculation one element at a time:

  int component = ef.getComponent();
  for ( ; lhs != endi, rhs1 != endi, rhs2 != endi;
	++lhs, ++rhs1, ++rhs2) {
    (*lhs)[component] =
      ((*rhs2)[component] - (*rhs1)[component])*slopeMultipplier +
      (*rhs1)[component];
  }

}


// ----------------------------------------------------------------------------
// This type of boundary condition (linear extrapolation) does very much the
// same thing for any centering; Doesn't seem to be a need for specializations
// based on centering.
// ----------------------------------------------------------------------------

template<class T, unsigned D, class M, class C>
void ComponentLinearExtrapolateFaceBCApply(ComponentLinearExtrapolateFace
					   <T,D,M,C>& ef,
					   Field<T,D,M,C>& A )
{

  // Find the slab that is the destination.
  // That is, in English, get an NDIndex spanning elements in the guard layers
  // on the face associated with the ComponentLinearExtrapaloteFace object:

  const NDIndex<D>& domain( A.getDomain() ); // Spans whole Field
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());

  // The direction (dimension of the Field) associated with the active face.
  // The numbering convention makes this division by two return the right
  // value, which will be between 0 and (D-1):

  unsigned d = ef.getFace()/2;

  // Must loop explicitly over the number of guard layers:
  int nGuardLayers;

  // The following bitwise AND logical test returns true if ef.m_face is odd
  // (meaning the "high" or "right" face in the numbering convention) and
  // returns false if ef.m_face is even (meaning the "low" or "left" face in
  // the numbering convention):

  if (ef.getFace() & 1) {

    // For "high" face, index in active direction goes from max index of
    // Field plus 1 to the same plus number of guard layers:
    nGuardLayers = A.rightGuard(d);

  } else {

    // For "low" face, index in active direction goes from min index of
    // Field minus the number of guard layers (usually a negative number)
    // to the same min index minus 1 (usually negative, and usually -1):
    nGuardLayers = A.leftGuard(d);

  }

  // Loop over the number of guard layers, treating each layer as a separate
  // slab (contrast this with all other BC types, where all layers are a single
  // slab):
  for (int guardLayer = 1; guardLayer <= nGuardLayers; guardLayer++) {

    // For linear extrapolation, the multipplier increases with more distant
    // guard layers:
    int slopeMultipplier = -1*guardLayer;

    if (ef.getFace() & 1) {
      slab[d] = Index(domain[d].max() + guardLayer,
		      domain[d].max() + guardLayer);
    } else {
      slab[d] = Index(domain[d].min() - guardLayer,
		      domain[d].min() - guardLayer);
    }

    // Loop over all the LField's in the Field A:

    typename Field<T,D,M,Cell>::iterator_if fill_i;
    for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i) {

      // Cache some things we will use often below.

      // Pointer to the data for the current LField:
      LField<T,D> &fill = *(*fill_i).second;

      // NDIndex spanning all elements in the LField, including the guards:
      const NDIndex<D> &fill_alloc = fill.getAllocated();

      // If the previously-created boundary guard-layer NDIndex "slab"
      // contains any of the elements in this LField (they will be guard
      // elements if it does), assign the values into them here by applying
      // the boundary condition:

      if (slab.touches(fill_alloc)) {

	// Find what it touches in this LField.
	NDIndex<D> dest = slab.intersect(fill_alloc);

	// For linear extrapolation boundary conditions, the boundary
	// guard-layer elements are filled based on a slope and intercept
	// derived from two layers of interior values; the src1 and src2
	// NDIndexes specify these two interior layers, which are operated on
	// by mathematical operations whose results are put into dest.  The
	// ordering of what is defined as src1 and src2 is set differently for
	// hi and lo faces, to make the sign for extrapolation work out right:
	NDIndex<D> src1 = dest;
	NDIndex<D> src2 = dest;
	if (ef.getFace() & 1) {
	  src2[d] = Index(domain[d].max() - 1, domain[d].max() - 1, 1);
	  src1[d] = Index(domain[d].max(), domain[d].max(), 1);
	} else {
	  src1[d] = Index(0,0,1); // Note: hardwired to 0-base, stride-1; could
	  src2[d] = Index(1,1,1); //  generalize with domain.min(), etc.
	}

	// Assume that src1 and src2 are contained withi nthe fill_alloc LField
	// domain; I think this is always true if the vnodes are always at
	// least one guard-layer-width wide in number of physical elements:

	ComponentLinearExtrapolateFaceBCApply2(dest, src1, src2, fill, ef,
					       slopeMultipplier);

      }
    }
  }
}


//----------------------------------------------------------------------

template<class T, unsigned D, class M, class C>
PatchBC<T,D,M,C>::
PatchBC(unsigned face)
  : BCondBase<T,D,M,C>(face)
{


}

//-----------------------------------------------------------------------------
// PatchBC::apply(Field)
//
// Loop over the patches of the Field, and call the user supplied
// apply(Field::iterator) member function on each.
//-----------------------------------------------------------------------------

template<class T, unsigned D, class M, class C>
void PatchBC<T,D,M,C>::apply( Field<T,D,M,C>& A )
{



  //------------------------------------------------------------
  // Find the slab that is the destination.
  //------------------------------------------------------------

  //
  // Get the physical domain for A.
  //
  const NDIndex<D>& domain( A.getDomain() );

  //
  // Start the calculation for the slab we'll be filling by adding
  // guard cells to the domain of A.
  //
  NDIndex<D> slab = AddGuardCells(domain,A.getGuardCellSizes());

  //
  // Get which direction is perpendicular to the face we'll be
  // filling.
  //
  unsigned d = this->getFace()/2;

  //
  // If the face is odd, we're looking at the 'upper' face, and if it
  // is even we're looking at the 'lower'.  Calculate the Index for
  // the dimension d.
  //
  if ( this->getFace() & 1 )
    slab[d] = Index( domain[d].max() + 1,  domain[d].max() + A.rightGuard(d));
  else
    slab[d] = Index( domain[d].min() - A.leftGuard(d), domain[d].min()-1 );

  //
  // Loop over all the vnodes.  We'll take action if it touches the slab.
  //
  int lfindex = 0;
  typename Field<T,D,M,C>::iterator_if fill_i;
  typename Field<T,D,M,C>::iterator p = A.begin();
  for (fill_i=A.begin_if(); fill_i!=A.end_if(); ++fill_i, ++lfindex)
    {
      //
      // Cache some things we will use often below.
      // fill: the current lfield.
      // fill_alloc: the total domain for that lfield.
      //
      LField<T,D> &fill = *(*fill_i).second;
      const NDIndex<D> &fill_alloc = fill.getAllocated();
      const NDIndex<D> &fill_owned = fill.getOwned();

      //
      // Is this an lfield we care about?
      //
      if ( slab.touches( fill_alloc ) )
        {
	  // need to worry about compression here
	  fill.Uncompress();

	  //
          // Find what part of this lfield is in the slab.
	  //
          NDIndex<D> dest = slab.intersect( fill_alloc );

	  //
	  // Set the iterator to the right spot in the Field.
	  //
	  vec<int,D> v;
	  for (int i=0; i<D; ++i)
	    v[i] = dest[i].first();
	  p.SetCurrentLocation( FieldLoc<D>(v,lfindex) );

	  //
	  // Let the user function do its stuff.
	  //
	  applyPatch(p,dest);
        }
    }
}

//----------------------------------------------------------------------

#undef COMPONENT_APPLY_BUILTIN

