// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 * This program was prepared by PSI. 
 * All rights in the program are reserved by PSI.
 * Neither PSI nor the author(s)
 * makes any warranty, express or implied, or assumes any liability or
 * responsibility for the use of this software
 *
 * Visit www.amas.web.psi for more details
 *
 ***************************************************************************/

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
#include "Field/GuardCellSizes.h"


template<unsigned Dim>
GuardCellSizes<Dim>::GuardCellSizes(unsigned s)
{
  
  
  for (unsigned d=0; d<Dim; ++d)
    Left[d] = Right[d] = s;
}

template<unsigned Dim>
GuardCellSizes<Dim>::GuardCellSizes(unsigned *s)
{
  
  
  for (unsigned d=0; d<Dim; ++d)
    Left[d] = Right[d] = s[d];
}

template<unsigned Dim>
GuardCellSizes<Dim>::GuardCellSizes(unsigned l, unsigned r)
{
  
  
  for (unsigned d=0; d<Dim; ++d) {
    Left[d] = l;
    Right[d] = r;
  }
}

template<unsigned Dim>
GuardCellSizes<Dim>::GuardCellSizes(unsigned *l, unsigned *r)
{
  
  
  for (unsigned d=0; d<Dim; ++d) {
    Left[d] = l[d];
    Right[d] = r[d];
  }
}

template<unsigned Dim>
void GuardCellSizes<Dim>::set_Left(unsigned s)
{
  
  
  for (unsigned d=0; d<Dim; ++d)
    Left[d] = s;
}

template<unsigned Dim>
void GuardCellSizes<Dim>::set_Left(unsigned *s)
{
  
  
  for (unsigned d=0; d<Dim; ++d)
    Left[d] = s[d];
}

template<unsigned Dim>
void GuardCellSizes<Dim>::set_Left(unsigned d, unsigned *s)
{
  
  
  Left[d] = s[d];
}

template<unsigned Dim>
void GuardCellSizes<Dim>::set_Right(unsigned s)
{
  
  
  for (unsigned d=0; d<Dim; ++d)
    Right[d] = s;
}

template<unsigned Dim>
void GuardCellSizes<Dim>::set_Right(unsigned *s)
{
  
  
  for (unsigned d=0; d<Dim; ++d)
    Right[d] = s[d];
}

template<unsigned Dim>
void GuardCellSizes<Dim>::set_Right(unsigned d, unsigned *s)
{
  
  
  Right[d] = s[d];
}

//////////////////////////////////////////////////////////////////////

template<unsigned Dim>
std::ostream&
operator<<(std::ostream& out, const GuardCellSizes<Dim>& gc)
{
  
  
  for (unsigned d=0; d<Dim; ++d)
    out << "[" << gc.left(d) << "," << gc.right(d) << "]";
  return out;
}

/***************************************************************************
 * $RCSfile: GuardCellSizes.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:26 $
 * IPPL_VERSION_ID: $Id: GuardCellSizes.cpp,v 1.1.1.1 2003/01/23 07:40:26 adelmann Exp $ 
 ***************************************************************************/
