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
		      e_dim_tag *p=0, 
		      int vnodes=-1);

  // Special constructor which uses a existing partition
  // particular from expde


  // Constructors for 1 ... 6 dimensions with parallel/serial specifiers:
  CenteredFieldLayout(Mesh& mesh, 
		      e_dim_tag p1, 
		      int vnodes=-1);
  CenteredFieldLayout(Mesh& mesh, 
		      e_dim_tag p1, e_dim_tag p2, 
		      int vnodes=-1);
  CenteredFieldLayout(Mesh& mesh, 
		      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, 
		      int vnodes=-1);
  CenteredFieldLayout(Mesh& mesh, 
		      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, e_dim_tag p4,
		      int vnodes=-1);
  CenteredFieldLayout(Mesh& mesh, 
		      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, e_dim_tag p4,
		      e_dim_tag p5,
		      int vnodes=-1);
  CenteredFieldLayout(Mesh& mesh, 
		      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, e_dim_tag p4,
		      e_dim_tag p5, e_dim_tag p6,
		      int vnodes=-1);
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // These specify both the total number of vnodes and the numbers of vnodes
  // along each dimension for the partitioning of the index space. Obviously
  // this restricts the number of vnodes to be a product of the numbers along
  // each dimension (the constructor implementation checks this):

  // Constructor for arbitrary dimension with parallel/serial specifier array:
  CenteredFieldLayout(Mesh& mesh, e_dim_tag *p, 
		      unsigned* vnodesAlongDirection, 
		      bool recurse=false,
		      int vnodes=-1);
  // Constructors for 1 ... 6 dimensions with parallel/serial specifiers:
  CenteredFieldLayout(Mesh& mesh,
		      e_dim_tag p1,
		      unsigned vnodes1,
		      bool recurse=false,
		      int vnodes=-1);
  CenteredFieldLayout(Mesh& mesh,
		      e_dim_tag p1, e_dim_tag p2,
		      unsigned vnodes1, unsigned vnodes2,
		      bool recurse=false,
		      int vnodes=-1);
  CenteredFieldLayout(Mesh& mesh,
		      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3,
		      unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		      bool recurse=false,
		      int vnodes=-1);
  CenteredFieldLayout(Mesh& mesh,
		      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, 
		      e_dim_tag p4,
		      unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		      unsigned vnodes4,
		      bool recurse=false,
		      int vnodes=-1);
  CenteredFieldLayout(Mesh& mesh,
		      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, 
		      e_dim_tag p4, e_dim_tag p5,
		      unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		      unsigned vnodes4, unsigned vnodes5,
		      bool recurse=false,
		      int vnodes=-1);
  CenteredFieldLayout(Mesh& mesh,
		      e_dim_tag p1, e_dim_tag p2, e_dim_tag p3, 
		      e_dim_tag p4, e_dim_tag p5, e_dim_tag p6,
		      unsigned vnodes1, unsigned vnodes2, unsigned vnodes3,
		      unsigned vnodes4, unsigned vnodes5, unsigned vnodes6,
		      bool recurse=false,
		      int vnodes=-1);
  //---------------------------------------------------------------------------

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

#endif // CENTERED_FIELD_LAYOUT_H

/***************************************************************************
 * $RCSfile: CenteredFieldLayout.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:27 $
 * IPPL_VERSION_ID: $Id: CenteredFieldLayout.h,v 1.1.1.1 2003/01/23 07:40:27 adelmann Exp $ 
 ***************************************************************************/
