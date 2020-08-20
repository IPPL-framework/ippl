// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef GUARD_CELL_SIZES_H
#define GUARD_CELL_SIZES_H

// include files
#include "Index/NDIndex.h"

#include <iostream>

template<unsigned Dim>
class GuardCellSizes
{

public:

  GuardCellSizes()
  {
    for (unsigned d=0; d<Dim; ++d)
      Left[d] = Right[d] = 0;
  }
  GuardCellSizes(unsigned s);
  GuardCellSizes(unsigned *s);
  GuardCellSizes(unsigned l, unsigned r);
  GuardCellSizes(unsigned *l, unsigned *r);
  constexpr GuardCellSizes<Dim>(const GuardCellSizes<Dim>&) = default;
  GuardCellSizes<Dim>& operator=(const GuardCellSizes<Dim>& gc)
  {
    for (unsigned d=0; d<Dim; ++d) {
      Left[d]  = gc.Left[d];
      Right[d] = gc.Right[d];
    }
    return *this;
  }

  void set_Left(unsigned s);
  void set_Left(unsigned *s);
  void set_Left(unsigned d, unsigned *s);

  void set_Right(unsigned s);
  void set_Right(unsigned *s);
  void set_Right(unsigned d, unsigned *s);

  unsigned left(unsigned d) const { return Left[d]; }
  unsigned right(unsigned d) const { return Right[d]; }

  // Lexigraphic compare of two GuardCellSizes so we can
  // use them as a Key in a map.
  bool operator<(const GuardCellSizes<Dim>& r) const ;      
  bool operator==(const GuardCellSizes<Dim>& r) const;

private:

  unsigned Left[Dim];
  unsigned Right[Dim];

};

template<unsigned Dim>
inline NDIndex<Dim>
AddGuardCells(const NDIndex<Dim>& idx, const GuardCellSizes<Dim>& g)
{
  NDIndex<Dim> ret;
  for (unsigned int d=0; d<Dim; ++d)
    ret[d] = Index(idx[d].min() - g.left(d), idx[d].max() + g.right(d));
  return ret;
}


template<unsigned Dim>
// Lexigraphic compare of two GuardCellSizes so we can
// use them as a Key in a map.
inline bool 
GuardCellSizes<Dim>::operator<(const GuardCellSizes<Dim>& r) const 
{
  for (unsigned d=0; d<Dim; ++d) {
    if ( left(d) != r.left(d) )
      return ( left(d) < r.left(d) );
    if ( right(d) != r.right(d) )
      return ( right(d) < r.right(d) );
  }
  // If we get here they're equal.
  return false;
}

template<unsigned Dim>
inline bool 
GuardCellSizes<Dim>::operator==(const GuardCellSizes<Dim>& r) const 
{
  for (unsigned d=0; d<Dim; ++d) {
    if ( left(d) != r.left(d) )
      return false;
    if ( right(d) != r.right(d) )
      return false;
  }
  // If we get here they're equal.
  return true;
}

//////////////////////////////////////////////////////////////////////

template<unsigned Dim>
std::ostream& operator<<(std::ostream&,const GuardCellSizes<Dim>&);

//////////////////////////////////////////////////////////////////////

#include "Field/GuardCellSizes.hpp"

#endif // GUARD_CELL_SIZES_H

/***************************************************************************
 * $RCSfile: GuardCellSizes.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: GuardCellSizes.h,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
