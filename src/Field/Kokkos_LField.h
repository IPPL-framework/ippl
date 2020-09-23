//
// Class Kokkos_LField
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
#ifndef KOKKOS_LField_H
#define KOKKOS_LField_H

#include <Kokkos_Core.hpp>

#include "Field/ViewTypes.h"

#include "Field/FieldExpr.h"

#include <iostream>


#include "Index/NDIndex.h"

// This stores the local data for a Field.
template<class T, unsigned Dim>
class Kokkos_LField : public FieldExpr<Kokkos_LField<T, Dim> >
{

public:
    // The type of domain stored here
    typedef NDIndex<Dim> Domain_t;

    typedef typename ViewType<T, Dim>::view_type view_type;


    /*! Ctor for an Kokkos_LField.  Arguments:
    * @param owned domain of "owned" region of Kokkos_LField (without guards)
    * @param vnode global vnode ID number
    */
    Kokkos_LField(const Domain_t& owned, int vnode = -1);

    // Copy constructor.
    Kokkos_LField(const Kokkos_LField<T,Dim>&);

    ~Kokkos_LField() {};

    template<typename ...Args>
    void resize(Args... args);

    //
    // General information accessors
    //

    // Return information about the Kokkos_LField.
    int size(unsigned d) const { return owned_m[d].length(); }
    const Domain_t& getOwned()       const { return owned_m; }

    // Return global vnode ID number (between 0 and nvnodes - 1)
    int getVnode() const { return vnode_m; }

    // print an Kokkos_LField out
    void write(std::ostream& out = std::cout) const;


    Kokkos_LField<T,Dim>& operator=(T x)
    {
        Kokkos::parallel_for("Kokkos_LField::operator=()",
                            dview_m.extent(0), KOKKOS_LAMBDA(const int i) {
                                dview_m(i) = x;
                        });
        Kokkos::fence();
        return *this;
    }


    Kokkos_LField<T,Dim>& operator=(const Kokkos_LField<T,Dim>& rhs)
    {
        Kokkos::parallel_for("Kokkos_LField::operator=()",
                            dview_m.extent(0), KOKKOS_LAMBDA(const int i) {
                                dview_m(i) = rhs.dview_m(i);
                        });
        Kokkos::fence();
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    T& operator() (size_t i) {
        return dview_m(i);
    }

    KOKKOS_INLINE_FUNCTION
    const T& operator() (size_t i) const {
        return dview_m(i);
    }

    template <typename E,
                std::enable_if_t<
    //               // essentially equivalent to:
    //               //   requires std::derived_from<E, VecExpr<E>>
                std::is_base_of_v<FieldExpr<E>, E>,
    //               // -------------------------------------------
                int> = 0>
    inline Kokkos_LField<T,Dim>& operator=(E const& expr) {
        Kokkos::parallel_for("Kokkos_LField<T,Dim>::operator=",
                                dview_m.extent(0), KOKKOS_LAMBDA(const int i) {
                                    dview_m(i) = expr(i);
                            });
        return *this;
    }

public:
    // actual field data
    view_type dview_m;

private:
    // Global vnode ID number for the associated Vnode (useful with more recent
    // FieldLayouts which store a logical "array" of vnodes; user specifies
    // numbers of vnodes along each direction). Classes or user codes that use
    // Kokkos_LField are responsible for setting and managing the values of this index;
    // if unset, it has the value -1. Generally, this parameter value is set on
    // construction of the vnode:

    int vnode_m;

    // What domain in the data is owned by this Kokkos_LField.

    Domain_t   owned_m;

    Kokkos_LField() = delete;
};

//////////////////////////////////////////////////////////////////////

#include "Field/Kokkos_LField.hpp"

#endif