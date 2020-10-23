//
// Class LField
//   Local fields. A BareField consists of mulitple LFields.
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
#ifndef IPPL_LField_H
#define IPPL_LField_H

#include <Kokkos_Core.hpp>

#include "Field/ViewTypes.h"

#include "Expression/IpplExpressions.h"

#include <iostream>
#include <cinttypes>

#include "Index/NDIndex.h"

namespace ippl {
    /*!
     * @file LField.h
     *
     * Local fields which store the actual field data.
     */

    /*!
     * @class LField
     * @tparam T data type
     * @tparam Dim field dimension
     * @warning The implementation currently only supports 3-dimensional fields. The reason are runtime issues
     * with "if constrexpr" in the assignment operator when running on GPU.
     */
    template<class T, unsigned Dim>
    class LField : public Expression<LField<T, Dim>, sizeof(typename detail::ViewType<T, Dim>::view_type)>
    {
    public:
        //! Domain type specifying the index region
        typedef NDIndex<Dim> Domain_t;

        //! View type storing the data
        typedef typename detail::ViewType<T, Dim>::view_type view_type;


        /*! Constructor for an LField. The default constructor is deleted.
        * @param owned domain of "owned" region of LField (without guards)
        * @param vnode global vnode ID number
        * @param nghost number of ghost layers
        */
        LField(const Domain_t& owned, int vnode = -1, int nghost = 1);

        //! Copy constructor.
        LField(const LField<T,Dim>&) = default;

        //! Default destructor.
        ~LField() = default;

        /*!
         * Dimension independent view resize function which calls Kokkos.
         * @tparam Args... variadic template specifying the individiual
         * dimension arguments
         */
        template<typename ...Args>
        void resize(Args... args);

        //
        // General information accessors
        //

        /*!
         * Local field size.
         * @param d the dimension
         * @returns the number of grid points in the given dimension.
         */
        int size(unsigned d) const { return owned_m[d].length(); }


        /*!
         * Index domain of the local field.
         * @returns the index domain.
         */
        const Domain_t& getOwned()       const { return owned_m; }

        /*!
         * @returns the global vnode ID number (between 0 and nvnodes - 1)
         */
        int getVnode() const { return vnode_m; }

        /*!
         * Print the LField.
         * @param out stream
         */
        void write(std::ostream& out = std::cout) const;

        /*!
         * Assign the same value to the whole field.
         */
        LField<T,Dim>& operator=(T x);

        /*!
         * Assign another local field.
         */
        LField<T,Dim>& operator=(const LField<T,Dim>&) = default;

        /*!
         * Assign another local field.
         * @tparam Args... variadic template to specify an access index for
         * a view element.
         * @param args view indices
         * @returns a view element
         */
        template<typename ...Args>
        KOKKOS_INLINE_FUNCTION
        T operator() (Args... args) const {
            return dview_m(args...);
        }


        view_type getView() {
            return dview_m;
        }

        /*!
         * Assign an arbitrary LField expression
         * @tparam E expression type
         * @tparam N size of the expression, this is necessary for running on the
         * device since otherwise it does not allocate enough memory
         * @param expr is the expression
         */
        template <typename E, size_t N>/*,
                    std::enable_if_t<
        //               // essentially equivalent to:
        //               //   requires std::derived_from<E, VecExpr<E>>
                    std::is_base_of_v<Expression<E, N>, E>,
        //               // -------------------------------------------
                    int> = 0>*/
        LField<T,Dim>& operator=(Expression<E, N> const& expr);

        T sum();
        T max();
        T min();
        T prod();

    private:
        // Global vnode ID number for the associated Vnode (useful with more recent
        // FieldLayouts which store a logical "array" of vnodes; user specifies
        // numbers of vnodes along each direction). Classes or user codes that use
        // LField are responsible for setting and managing the values of this index;
        // if unset, it has the value -1. Generally, this parameter value is set on
        // construction of the vnode:

        //!
        int vnode_m;

        //! Number of ghost layers on each field boundary
        int nghost_m;

        //! Actual field data
        view_type dview_m;

        //! Domain of the data
        Domain_t owned_m;

        LField() = delete;
    };
}

//////////////////////////////////////////////////////////////////////

#include "Field/LField.hpp"

#endif
