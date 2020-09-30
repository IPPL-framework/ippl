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
#include "Field/Kokkos_BareField.h"
#include "Field/BrickExpression.h"
#include "FieldLayout/FieldLayout.h"
#include "Message/Communicate.h"
#include "Message/GlobalComm.h"
#include "Message/Tags.h"
#include "Utility/Inform.h"
#include "Utility/Unique.h"
#include "Utility/IpplInfo.h"
#include "Utility/IpplStats.h"

#include <map>
#include <utility>
#include <cstdlib>


template< class T, unsigned Dim >
Kokkos_BareField<T,Dim>::~Kokkos_BareField() { }


template< class T, unsigned Dim >
void
Kokkos_BareField<T,Dim>::initialize(Layout_t & l) {
  
  

  // if our Layout has been previously set, we just ignore this request
  if (Layout == 0) {
    Layout = &l;
    setup();
  }
}

//////////////////////////////////////////////////////////////////////
//
// Using the data that has been initialized by the ctors,
// complete the construction by allocating the Kokkos_LFields.
//

template< class T, unsigned Dim >
void
Kokkos_BareField<T,Dim>::setup()
{
  // Loop over all the Vnodes, creating an Kokkos_LField in each.
  for (typename Layout_t::iterator_iv v_i=getLayout().begin_iv();
       v_i != getLayout().end_iv();
       ++v_i)
    {
      // Get the owned.
      const NDIndex<Dim> &owned = (*v_i).second->getDomain();

      // Get the global vnode number (ID number, value from 0 to nvnodes-1):
      int vnode = (*v_i).second->getVnode();

      // Put it in the list.
      lfields_m.push_back(LField_t(owned, vnode));
    }
}

//////////////////////////////////////////////////////////////////////

//
// Print a Kokkos_BareField out.
//

template< class T, unsigned Dim>
void 
Kokkos_BareField<T,Dim>::write(std::ostream& out)
{
    for (const auto& lf : lfields_m) {
        lf.write(out);
    }
}




template <typename E1, typename E2>
class BareFieldSum: public BareFieldExpr<BareFieldSum<E1, E2> >{

public:
  BareFieldSum(E1 const& u, E2 const& v) : _u(u), _v(v) { }

  auto operator()(size_t i) const { return _u(i) + _v(i); }

private:
  E1 const& _u;
  E2 const& _v;
};

template <typename E1, typename E2>
BareFieldSum<E1, E2>
operator+(BareFieldExpr<E1> const& u, BareFieldExpr<E2> const& v) {
  return BareFieldSum<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}



template <typename E1, typename E2>
class BareFieldSubtract: public BareFieldExpr<BareFieldSubtract<E1, E2> >{

public:
  BareFieldSubtract(E1 const& u, E2 const& v) : _u(u), _v(v) { }

  auto operator()(size_t i) const { return _u(i) - _v(i); }

private:
  E1 const& _u;
  E2 const& _v;
};

template <typename E1, typename E2>
BareFieldSubtract<E1, E2>
operator-(BareFieldExpr<E1> const& u, BareFieldExpr<E2> const& v) {
  return BareFieldSubtract<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}


template <typename E1, typename E2>
class BareFieldMultiply: public BareFieldExpr<BareFieldMultiply<E1, E2> >{

public:
  BareFieldMultiply(E1 const& u, E2 const& v) : _u(u), _v(v) { }

  auto operator()(size_t i) const { return _u(i) * _v(i); }

private:
  E1 const& _u;
  E2 const& _v;
};

template <typename E1, typename E2>
BareFieldMultiply<E1, E2>
operator*(BareFieldExpr<E1> const& u, BareFieldExpr<E2> const& v) {
  return BareFieldMultiply<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}


template <typename E1, typename E2>
class BareFieldDivide: public BareFieldExpr<BareFieldDivide<E1, E2> >{

public:
  BareFieldDivide(E1 const& u, E2 const& v) : _u(u), _v(v) { }

  auto operator()(size_t i) const { return _u(i) / _v(i); }

private:
  E1 const& _u;
  E2 const& _v;
};

template <typename E1, typename E2>
BareFieldDivide<E1, E2>
operator/(BareFieldExpr<E1> const& u, BareFieldExpr<E2> const& v) {
  return BareFieldDivide<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));

}

// //////////////////////////////////////////////////////////////////////
// // Get a ref to a single element of the Field; if it is not local to our
// // processor, print an error and exit.  This allows the user to provide
// // different index values on each node, instead of using the same element
// // and broadcasting to all nodes.
// template<class T, unsigned Dim>
// T/*&*/ Kokkos_BareField<T,Dim>::localElement(const NDIndex<Dim>& Indexes) /*const*/
// {
// /*
//
//
//
//   // Instead of checking to see if the user has asked for one element,
//   // we will just use the first element specified for each dimension.
//
//   // Is this element here?
//   // Try and find it in the local Kokkos_BareFields.
//   const_iterator_if lf_i   = begin_if();
//   const_iterator_if lf_end = end_if();
//   for ( ; lf_i != lf_end ; ++lf_i ) {
//     Kokkos_LField<T,Dim>& lf(*(*lf_i).second);
//     // End-point "contains" OK since "owned" is unit stride.
//     // was before CK fix: if ( lf.getOwned().contains( Indexes ) ) {
//     if ( lf.getAllocated().contains( Indexes ) ) {
//       // Found it ... first uncompress, then get a pointer to the
//       // requested element.
//       lf.Uncompress();
//       //      return *(lf.begin(Indexes));
//       // instead of building an iterator, just find the value
//       NDIndex<Dim> alloc = lf.getAllocated();
//       T* pdata = PtrOffset(lf.getP(), Indexes, alloc,
//                            Kokkos_LFieldDimTag<Dim,(Dim<=3)>());
//       return *pdata;
//     }
//   }
//
//   // if we're here, we did not find it ... it must not be local
//   ERRORMSG("Kokkos_BareField::localElement: attempt to access non-local index ");
//   ERRORMSG(Indexes << " on node " << Ippl::myNode() << endl);
//   ERRORMSG("Occurred in a Kokkos_BareField with layout = " << getLayout() << endl);
//   ERRORMSG("Calling abort ..." << endl);
//   Ippl::abort();
//   return *((*((*(begin_if())).second)).begin());*/
//
//     return double(2.0);
// }
