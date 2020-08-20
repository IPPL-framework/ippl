// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef RNODE_H
#define RNODE_H

/***********************************************************************
 * Rnodes really have very little information.
 * They know the region of the Rnode, the physical node it resides on,
 * and the offset (if any) for the node if it is a periodic image of
 * the 'real' Rnode.
 ***********************************************************************/

// include files
#include "Region/NDRegion.h"
#include "AppTypes/Vektor.h"

// forward declarations
template <unsigned Dim> class NDIndex;


template<class T,unsigned Dim>
class Rnode {

public: 
  // constructors
  Rnode(const NDRegion<T,Dim>& domain, const Vektor<T,Dim>& offset, int node)
    : Domain(domain), Offset(offset), Node(node) {}
  Rnode(const NDRegion<T,Dim>& domain, int node)
    : Domain(domain), Node(node) {}
  Rnode(const NDIndex<Dim>& domain, const Vektor<T,Dim>& offset, int node)
    : Domain(domain), Offset(offset), Node(node) {}
  Rnode(const NDIndex<Dim>& domain, int node)
    : Domain(domain), Node(node) {}

  // operator=
  Rnode<T,Dim>& operator=(const Rnode<T,Dim>& r) {
    Domain = r.Domain;
    Offset = r.Offset;
    Node   = r.Node;
    return *this;
  }

  // query functions
  int getNode() { return Node; }
  const NDRegion<T,Dim>& getDomain() { return Domain; }
  const Vektor<T,Dim>& getOffset() { return Offset; }
  
private:
  NDRegion<T,Dim> Domain;
  Vektor<T,Dim>   Offset;
  int Node;
};

#endif // RNODE_H

/***************************************************************************
 * $RCSfile: Rnode.h,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:32 $
 * IPPL_VERSION_ID: $Id: Rnode.h,v 1.1.1.1 2003/01/23 07:40:32 adelmann Exp $ 
 ***************************************************************************/
