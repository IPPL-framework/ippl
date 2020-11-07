// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef CENTERED_FIELD_LAYOUT_H
#define CENTERED_FIELD_LAYOUT_H

// include files
#include "FieldLayout/FieldLayout.h"


template<unsigned Dim, class Mesh, class Centering>
class CenteredFieldLayout : public FieldLayout<Dim>
{
public:
  //---------------------------------------------------------------------------
  // Constructors from a mesh object only and parallel/serial specifiers.
  // If not doing this, user should be just using simple FieldLayout object, 
  // though no harm would be done in constructiong a CenteredFieldLayout with
  // Index/NDIndex arguments via the inherited constructors from FieldLayout.
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // These specify only a total number of vnodes, allowing the constructor
  // complete control on how to do the vnode partitioning of the index space:
  // Constructor for arbitrary dimension with parallel/serial specifier array:

  // This one also works if nothing except mesh is specified:
  CenteredFieldLayout(Mesh& mesh, 
		      e_dim_tag *p=0);

  //---------------------------------------------------------------------------
  // A constructor a a completely user-specified partitioning of the
  // mesh space.

  CenteredFieldLayout(Mesh& mesh,
		      const NDIndex<Dim> *dombegin,
		      const NDIndex<Dim> *domend,
		      const int *nbegin,
		      const int *nend);
};

#include "FieldLayout/CenteredFieldLayout.hpp"

#endif
