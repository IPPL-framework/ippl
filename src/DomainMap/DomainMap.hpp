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
#include "Index/NDIndex.h"
#include "Field/LField.h"
#include "FieldLayout/Vnode.h"






//////////////////////////////////////////////////////////////////////

// Insert a new key in the DomainMap.

template < class Key , class T , class Touches, class Contains, class Split > 
void
DomainMap<Key,T,Touches,Contains,Split>::insert(const value_type& d,
						bool noSplit)
{
   
  

  // Tell the tree top to insert.
  Root->insert(d, noSplit);
  // Increment the total size.
  Size += 1;
  // Update the pointer to the first element.
  update_leftmost();
}

//////////////////////////////////////////////////////////////////////

// Recalculate the first element in the tree.

template < class Key , class T , class Touches, class Contains, class Split > 
void
DomainMap<Key,T,Touches,Contains,Split>::update_leftmost()
{
   
  

  Node *p=Root;
  // First dive all the way left.
  while ( p->Left )
    p = p->Left;
  // Look for the first element.
  typename Node::cont_type::iterator v;
  while ( (v=p->cont.begin()) == p->cont.end() ) {
    // First look right.
    if ( p->Right ) {
      // If it is there, go there and then all the way left.
      p = p->Right;
      while ( p->Left )
	p = p->Left;
    }
    else {
      // If there is no right, go up until you can go right more.
      Node *y = p->Parent;
      while (y && (p==y->Right)) {
	p = y;
	y = y->Parent;
      }
      p = y;
    }
  }
  if ( (Leftmost.p=p) != 0 ) 
    Leftmost.v = v;
}

//////////////////////////////////////////////////////////////////////

// Find the range that touches a given T.

template < class Key, class T , class Touches, class Contains, class Split > 
std::pair<typename DomainMap<Key,T,Touches,Contains,Split>::touch_iterator,
          typename DomainMap<Key,T,Touches,Contains,Split>::touch_iterator>
DomainMap<Key,T,Touches,Contains,Split>::touch_range(const Key& t) const
{ 

	Node *p=Root;

	if ( p ) {

		// First dive left, checking touches.
		for ( Node* y=p->Left; y && Touches::test(t,y->MyDomain); y=y->Left )
			p = y;

		do {
			// Look for the first element that actually touches.
			for (typename Node::cont_type::iterator v=p->cont.begin();
					v!=p->cont.end(); ++v)
				if ( Touches::test(t,(*v).first) ) {
					// Found it!
					touch_iterator f;
					f.TouchThis = t;
					f.p = p;
					f.v = v;
					return std::pair<touch_iterator,touch_iterator>(f,touch_iterator());
				}

			// Didn't find one here. Move on.
			Node *y = p->Right;
			if ( y && Touches::test(t,y->MyDomain) ) {
				// If it is there, go there and then all the way left.
				p = y;
				for ( y=p->Left; y && Touches::test(t,y->MyDomain) ; y=y->Left )
					p = y;
			}
			else {
				// If there is no right, go up until you can go right more.
				for ( y = p->Parent; y && (p==y->Right); y = y->Parent )
					p = y;
				p = y;
			}
		} while (p);
	}
	// Didn't find any.
	return std::pair<touch_iterator,touch_iterator>(touch_iterator(),touch_iterator());
}

//////////////////////////////////////////////////////////////////////

//
// Copy ctor and assign do deep copies.
//

template < class Key, class T , class Touches, class Contains, class Split > 
DomainMap<Key,T,Touches,Contains,Split>::
DomainMap(const DomainMap<Key,T,Touches,Contains,Split>& a)
: Root( new Node(a.Root->MyDomain) ), Size(0)
{
   
  

  for (iterator p=a.begin(); p!=a.end(); ++p)
    insert_noupdate( *p );
  update_leftmost();
}

template < class Key, class T , class Touches, class Contains, class Split > 
void
DomainMap<Key,T,Touches,Contains,Split>::
operator=(const DomainMap<Key,T,Touches,Contains,Split>& a)
{
   
  

  if ( this != &a ) {
    // Clean out the current contents.
    delete Root;
    Root = new Node( a.Root->MyDomain );
    Size = 0;
    // Add the new stuff.
    for (iterator p=a.begin(); p!=a.end(); ++p)
      insert_noupdate( *p );
    // Reset the pointer to the first one.
    update_leftmost();
  }
}
/***************************************************************************
 * $RCSfile: DomainMap.cpp,v $   $Author: adelmann $
 * $Revision: 1.1.1.1 $   $Date: 2003/01/23 07:40:25 $
 * IPPL_VERSION_ID: $Id: DomainMap.cpp,v 1.1.1.1 2003/01/23 07:40:25 adelmann Exp $ 
 ***************************************************************************/
