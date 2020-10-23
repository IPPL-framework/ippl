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
#ifndef IPPL_BARE_FIELD_H
#define IPPL_BARE_FIELD_H

#include "Field/LField.h"
#include "Utility/IpplInfo.h"
#include "Utility/PAssert.h"

#include <iostream>
#include <cstdlib>

// forward declarations
class Index;
template<unsigned Dim> class NDIndex;
template<unsigned Dim> class FieldLayout;

namespace ippl {
    /*!
     * @file BareField.h
     * A BareField represents a real field. It may consist of multiple
     * local fields.
     */

    /*!
     * @class BareField
     * @tparam T data type
     * @tparam Dim field dimension
     */
    template<typename T,  unsigned Dim>
    class BareField : public FieldExpression< BareField<T, Dim> >
    {

    public:
        typedef FieldLayout<Dim> Layout_t;
        typedef LField<T, Dim> LField_t;
        typedef std::deque<std::shared_ptr<LField_t>> container_t;

        typedef typename container_t::iterator iterator_t;

        // A default constructor, which should be used only if the user calls the
        // 'initialize' function before doing anything else.  There are no special
        // checks in the rest of the BareField methods to check that the field has
        // been properly initialized.
        BareField();

        BareField(Layout_t&);

        BareField(const BareField&) = default;

        // Destroy the BareField.
        ~BareField() = default;

        // Initialize the field, if it was constructed from the default constructor.
        // This should NOT be called if the field was constructed by providing
        // a FieldLayout.
        void initialize(Layout_t&);


        iterator_t begin() noexcept {
            return lfields_m.begin();
        }

        iterator_t end() noexcept {
            return lfields_m.end();
        }


        LField_t& operator()(size_t i) {
            return *lfields_m[i];
        }

        const LField_t& operator()(size_t i) const {
            return *lfields_m[i];
        }


        const LField_t& operator[](size_t i) const {
            return *lfields_m[i];
        }


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

        template <typename E>
        BareField<T, Dim>& operator=(const FieldExpression<E>& expr);

        void write(std::ostream& = std::cout);


    protected:
        //! Container to store the local fields
        container_t lfields_m;

    private:
        /*!
         * Allocate all the local fields.
         */
        void setup();

        //! How the local arrays are laid out.
        Layout_t* layout_m;
    };
}

//////////////////////////////////////////////////////////////////////

#include "Field/BareField.hpp"

#endif
