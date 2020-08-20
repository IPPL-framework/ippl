// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef SUB_PARTICLE_ASSIGN_DEFS_H
#define SUB_PARTICLE_ASSIGN_DEFS_H

// include files
#include "SubField/SubFieldAssignDefs.h"
#include "SubParticle/SubParticleAssign.h"

//////////////////////////////////////////////////////////////////////
//
// Is the domain specification object compressed?
//
//////////////////////////////////////////////////////////////////////

template<class PA, class T, unsigned D, class C>
inline bool
for_each(SubParticleAttribIter<PA,T,D> &, DomainCompressed, C)
{
  return false;
}

//////////////////////////////////////////////////////////////////////
//
// Do the terms all use the same kind of subset object?
//
//////////////////////////////////////////////////////////////////////

template<class PA, class T, unsigned D, class C>
inline bool
for_each(SubParticleAttribIter<PA,T,D> &p,SameSubsetType s,C)
{
  return p.matchType(s.fID);
}

//////////////////////////////////////////////////////////////////////
//
// Initialize all subset objects in an expression before the loop starts
//
//////////////////////////////////////////////////////////////////////

template<class PA, class T, unsigned D, class C>
inline int
for_each(SubParticleAttribIter<PA,T,D> &p, SubsetInit, C)
{
  p.initialize();
  return 0;
}

//////////////////////////////////////////////////////////////////////
//
// Set a subfield iterator to point to the next lfield
//
//////////////////////////////////////////////////////////////////////

template<class PA, class T, unsigned D, class C>
inline int
for_each(SubParticleAttribIter<PA,T,D> &p,SubsetNextLField,C)
{
  p.nextLField();
  return 0;
}

//////////////////////////////////////////////////////////////////////
//
// Do any of the terms in an expression have an ID equal to a given one?
//
//////////////////////////////////////////////////////////////////////

template<class PA, class T, unsigned D, class C>
inline bool
for_each(SubParticleAttribIter<PA,T,D> &, SameFieldID, C)
{
  return false;
}

//////////////////////////////////////////////////////////////////////
//
// Plugbase.
//
//////////////////////////////////////////////////////////////////////

template<class PA, class T, unsigned D, class C>
inline bool
for_each(SubParticleAttribIter<PA,T,D> &p, const PlugBase<D>& f, C)
{
  return p.plugBase(f.Domain);
}

template<class PA, class T, unsigned D, class C>
inline bool
for_each(SubParticleAttribIter<PA,T,D> &, IsCompressed, C)
{
  return false;
}

//////////////////////////////////////////////////////////////////////
//
// Evaluation functors.
// Just need EvalFunctor_1 here.  EvalFunctor_0 is defined since we
// need it to compile, but it should never be called because it is
// only used when things are compressed, and SubParticleAttrib's are
// never compressed.
//
//////////////////////////////////////////////////////////////////////

template<class PA, class T, unsigned D>
inline T&
for_each(SubParticleAttribIter<PA,T,D> &p, const EvalFunctor_1 &e)
{
  return p.offset(e.I);
}

template<class PA, class T, unsigned D>
inline T&
for_each(SubParticleAttribIter<PA,T,D> &p, const EvalFunctor_0 &)
{
  // we should never be here
  ERRORMSG("SubParticleAttrib::iterator -> EvalFunctor_0 called.");
  Ippl::abort();

  // this is here just so we can have some kind of return value
  return p.offset(0);
}

//////////////////////////////////////////////////////////////////////
//
// Does an iterator reference something with unit stride?
//
//////////////////////////////////////////////////////////////////////

template<class PA, class T, unsigned D, class C>
inline bool
for_each(SubParticleAttribIter<PA,T,D> &/*p*/, HasUnitStride, C)
{
  return true;
}


//////////////////////////////////////////////////////////////////////
//
// Ask each term to fill guard cells and compress itself
//
//////////////////////////////////////////////////////////////////////

template<class PA, class T, unsigned D, class C, class T1>
inline int
for_each(SubParticleAttribIter<PA,T,D> &, const FillGCIfNecessaryTag<D,T1> &, C)
{
  return 0;
}


#endif // SUB_PARTICLE_ASSIGN_DEFS_H

/***************************************************************************
 * $RCSfile: SubParticleAssignDefs.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:33 $
 * IPPL_VERSION_ID: $Id: SubParticleAssignDefs.h,v 1.1.1.1 2003/01/23 07:40:33 adelmann Exp $ 
 ***************************************************************************/
