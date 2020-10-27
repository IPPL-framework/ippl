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
// #include "Field/BrickExpression.h"
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
    template< typename T, unsigned Dim>
    BareField<T, Dim>::BareField() : layout_m(nullptr) { }


    template< typename T, unsigned Dim>
    BareField<T, Dim>::BareField(Layout_t& l) : layout_m(&l) {
        setup();
    }

    template<typename T, unsigned Dim>
    void BareField<T, Dim>::initialize(Layout_t& l) {
        if (layout_m == 0) {
            layout_m = &l;
            setup();
        }
    }


    /* Using the data that has been initialized by the ctors,
     * complete the construction by allocating the LFields.
     */
    template<typename T, unsigned Dim>
    void BareField<T, Dim>::setup() {
        // Loop over all the Vnodes, creating an LField in each.
        for (typename Layout_t::iterator_iv v_i=getLayout().begin_iv();
             v_i != getLayout().end_iv(); ++v_i)
        {
            // Get the owned.
            const NDIndex<Dim> &owned = (*v_i).second->getDomain();

            // Get the global vnode number (ID number, value from 0 to nvnodes-1):
            int vnode = (*v_i).second->getVnode();

            // Put it in the list.
            lfields_m.push_back(std::shared_ptr<LField_t>(new LField_t(owned, vnode)));
        }
    }


    template<typename T, unsigned Dim>
    BareField<T, Dim>& BareField<T, Dim>::operator=(T x) {
        for (auto& lf : lfields_m) {
            *lf = x;
        }
        return *this;
    }


    template<typename T, unsigned Dim>
    template <typename E>
    BareField<T, Dim>& BareField<T, Dim>::operator=(const FieldExpression<E>& expr) {
        for (size_t i = 0; i < lfields_m.size(); ++i) {
            *lfields_m[i] = expr[i];
        }
        return *this;
    }

    template<typename T, unsigned Dim>
    void BareField<T, Dim>::write(std::ostream& out) {
        for (const auto& lf : lfields_m) {
            lf->write(out);
        }
    }
    
    #define DefineFieldReduction(name, op)                     \
    template<typename T, unsigned Dim>                         \
    T BareField<T, Dim>::name(int nghost) {                    \
        T temp = lfields_m[0]->name(nghost);                         \
        for (size_t i = 1; i < lfields_m.size(); ++i) {        \
            T myVal = lfields_m[i]->name(nghost);              \
            op;                                                \
        }                                                      \
        return temp;                                           \
    }

    DefineFieldReduction(sum,  temp += myVal)
    DefineFieldReduction(max,  if(myVal > temp) temp = myVal)
    DefineFieldReduction(min,  if(myVal < temp) temp = myVal)
    DefineFieldReduction(prod, temp *= myVal)

}
