//
// Class BareField
//   A BareField consists of multple LFields and represents a field.
//
// Copyright (c) 2020, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#include "Field/BareField.h"
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

namespace ippl {

    template< class T, unsigned Dim >
    BareField<T,Dim>::~BareField() { }


    template< class T, unsigned Dim >
    void
    BareField<T,Dim>::initialize(Layout_t & l) {



    // if our Layout has been previously set, we just ignore this request
    if (Layout == 0) {
        Layout = &l;
        setup();
    }
    }

    //////////////////////////////////////////////////////////////////////
    //
    // Using the data that has been initialized by the ctors,
    // complete the construction by allocating the LFields.
    //

    template< class T, unsigned Dim >
    void
    BareField<T,Dim>::setup()
    {
    // Loop over all the Vnodes, creating an LField in each.
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
    // Print a BareField out.
    //

    template< class T, unsigned Dim>
    void
    BareField<T,Dim>::write(std::ostream& out)
    {
        for (const auto& lf : lfields_m) {
            lf.write(out);
        }
    }

    // //////////////////////////////////////////////////////////////////////
    // // Get a ref to a single element of the Field; if it is not local to our
    // // processor, print an error and exit.  This allows the user to provide
    // // different index values on each node, instead of using the same element
    // // and broadcasting to all nodes.
    // template<class T, unsigned Dim>
    // T/*&*/ BareField<T,Dim>::localElement(const NDIndex<Dim>& Indexes) /*const*/
    // {
    // /*
    //
    //
    //
    //   // Instead of checking to see if the user has asked for one element,
    //   // we will just use the first element specified for each dimension.
    //
    //   // Is this element here?
    //   // Try and find it in the local BareFields.
    //   const_iterator_if lf_i   = begin_if();
    //   const_iterator_if lf_end = end_if();
    //   for ( ; lf_i != lf_end ; ++lf_i ) {
    //     LField<T,Dim>& lf(*(*lf_i).second);
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
    //                            LFieldDimTag<Dim,(Dim<=3)>());
    //       return *pdata;
    //     }
    //   }
    //
    //   // if we're here, we did not find it ... it must not be local
    //   ERRORMSG("BareField::localElement: attempt to access non-local index ");
    //   ERRORMSG(Indexes << " on node " << Ippl::myNode() << endl);
    //   ERRORMSG("Occurred in a BareField with layout = " << getLayout() << endl);
    //   ERRORMSG("Calling abort ..." << endl);
    //   Ippl::abort();
    //   return *((*((*(begin_if())).second)).begin());*/
    //
    //     return double(2.0);
    // }
}