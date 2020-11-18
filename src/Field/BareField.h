//
// Class BareField
//   A BareField represents a field.
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
#ifndef IPPL_BARE_FIELD_H
#define IPPL_BARE_FIELD_H

#include <Kokkos_Core.hpp>

#include "Expression/IpplExpressions.h"
#include "Types/ViewTypes.h"

#include "FieldLayout/FieldLayout.h"

#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"

#include <iostream>
#include <cstdlib>


namespace ippl {
    class Index;

    /*!
     * @file BareField.h
     * A BareField represents a real field.
     */

    /*!
     * @class BareField
     * @tparam T data type
     * @tparam Dim field dimension
     * @warning The implementation currently only supports 3-dimensional fields. The reason are runtime issues
     * with "if constrexpr" in the assignment operator when running on GPU.
     */
    template<typename T,  unsigned Dim>
    class BareField : public detail::Expression<BareField<T, Dim>, sizeof(typename detail::ViewType<T, Dim>::view_type)>
    {

    public:
        using Layout_t = FieldLayout<Dim>;

        //! Domain type specifying the index region
        using Domain_t = typename Layout_t::NDIndex_t;

        //! View type storing the data
        using view_type = typename detail::ViewType<T, Dim>::view_type;
        using HostMirror = typename view_type::host_mirror_type;


        /*! A default constructor, which should be used only if the user calls the
         * 'initialize' function before doing anything else.  There are no special
         * checks in the rest of the BareField methods to check that the field has
         * been properly initialized.
         */
        BareField();

        /*! Constructor for a BareField. The default constructor is deleted.
         * @param l of field
         * @param nghost number of ghost layers
         */
        BareField(Layout_t& l, int nghost = 1);

        BareField(const BareField&) = default;

        // Destroy the BareField.
        ~BareField() = default;

        /*!
         * Dimension independent view resize function which calls Kokkos.
         * @tparam Args... variadic template specifying the individiual
         * dimension arguments
         */
        template<typename ...Args>
        void resize(Args... args);


        /*!
         * Initialize the field, if it was constructed from the default constructor.
         * This should NOT be called if the field was constructed by providing
         * a FieldLayout.
         * @param l of field
         * @param nghost number of ghost layers
         */
        void initialize(Layout_t& l, int nghost = 1);


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


        // Access to the layout.
        Layout_t &getLayout() const
        {
            PAssert(layout_m != 0);
            return *layout_m;
        }


        const Index& getIndex(unsigned d) const {return getLayout().getDomain()[d];}
        const NDIndex<Dim>& getDomain() const { return getLayout().getDomain(); }

        // Assignment from a constant.
        BareField<T, Dim>& operator=(T x);

        /*!
         * Assign an arbitrary BareField expression
         * @tparam E expression type
         * @tparam N size of the expression, this is necessary for running on the
         * device since otherwise it does not allocate enough memory
         * @param expr is the expression
         */
        template <typename E, size_t N>
        BareField<T, Dim>& operator=(const detail::Expression<E, N>& expr);

        /*!
         * Assign another field.
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


        view_type& getView() {
            return dview_m;
        }

        const view_type& getView() const {
            return dview_m;
        }

        HostMirror getHostMirror() {
            return Kokkos::create_mirror(dview_m);
        }

        /*!
         * Print the BareField.
         * @param out stream
         */
        void write(std::ostream& out = std::cout) const;

        T sum(int nghost = 0);
        T max(int nghost = 0);
        T min(int nghost = 0);
        T prod(int nghost = 0);


    private:
        //! Number of ghost layers on each field boundary
        int nghost_m;

        //! Actual field data
        view_type dview_m;

        //! Domain of the data
        Domain_t owned_m;

        /*!
         * Allocate field.
         */
        void setup();

        //! How the arrays are laid out.
        Layout_t* layout_m;
    };
}

//////////////////////////////////////////////////////////////////////

#include "Field/BareField.hpp"

#endif
