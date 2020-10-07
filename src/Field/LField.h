//
// Class LField
//   Local Field class
//
// Copyright (c) 2003 - 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of OPAL.
//
// OPAL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with OPAL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef IPPL_LField_H
#define IPPL_LField_H

#include <Kokkos_Core.hpp>

#include "Field/ViewTypes.h"

#include "Expression/IpplExpressions.h"

#include <iostream>
#include <cinttypes>

#include "Index/NDIndex.h"

namespace ippl {

    // This stores the local data for a Field.
    template<class T, unsigned Dim>
    class LField : public Expression<LField<T, Dim>, sizeof(typename ViewType<T, Dim>::view_type)>
    {
    public:
        typedef std::int64_t int64_t;
        // The type of domain stored here
        typedef NDIndex<Dim> Domain_t;

        typedef typename ViewType<T, Dim>::view_type view_type;


        /*! Ctor for an LField.  Arguments:
        * @param owned domain of "owned" region of LField (without guards)
        * @param vnode global vnode ID number
        */
        LField(const Domain_t& owned, int vnode = -1);

        // Copy constructor.
        LField(const LField<T,Dim>&) = default;

        ~LField() {};

        template<typename ...Args>
        void resize(Args... args);

        //
        // General information accessors
        //

        // Return information about the LField.
        int size(unsigned d) const { return owned_m[d].length(); }
        const Domain_t& getOwned()       const { return owned_m; }

        // Return global vnode ID number (between 0 and nvnodes - 1)
        int getVnode() const { return vnode_m; }

        // print an LField out
        void write(std::ostream& out = std::cout) const;

        LField<T,Dim>& operator=(T x);

        LField<T,Dim>& operator=(const LField<T,Dim>&) = default;

        template<typename ...Args>
        KOKKOS_INLINE_FUNCTION
        T operator() (Args... args) const {
            return dview_m(args...);
        }

        template <typename E, size_t N>/*,
                    std::enable_if_t<
        //               // essentially equivalent to:
        //               //   requires std::derived_from<E, VecExpr<E>>
                    std::is_base_of_v<Expression<E, N>, E>,
        //               // -------------------------------------------
                    int> = 0>*/
        LField<T,Dim>& operator=(Expression<E, N> const& expr);

    private:
        // Global vnode ID number for the associated Vnode (useful with more recent
        // FieldLayouts which store a logical "array" of vnodes; user specifies
        // numbers of vnodes along each direction). Classes or user codes that use
        // LField are responsible for setting and managing the values of this index;
        // if unset, it has the value -1. Generally, this parameter value is set on
        // construction of the vnode:

        int vnode_m;

        // actual field data
        view_type dview_m;

        // What domain in the data is owned by this LField.
        Domain_t   owned_m;

        LField() = delete;
    };
}

//////////////////////////////////////////////////////////////////////

#include "Field/LField.hpp"

#endif