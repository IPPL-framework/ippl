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
#include "Utility/Unique.h"
#include "Utility/my_auto_ptr.h"

#include <iostream>
#include <cstdlib>

// #include "Ippl/IpplExpressions.h"

// forward declarations
class Index;
template<unsigned Dim> class NDIndex;
template<unsigned Dim> class FieldLayout;

namespace ippl {

    // class definition
    template<typename T,  unsigned Dim>
    class BareField : public FieldExpression< BareField<T, Dim> >
    {

    public:
        // Some externally visible typedefs and enums
        typedef FieldLayout<Dim> Layout_t;
        typedef LField<T,Dim> LField_t;

        // A default constructor, which should be used only if the user calls the
        // 'initialize' function before doing anything else.  There are no special
        // checks in the rest of the BareField methods to check that the field has
        // been properly initialized.
        BareField();

        // Create a new BareField with a given layout and optional guard cells.
        BareField(Layout_t &);


        BareField(const BareField&) = default;

        // Destroy the BareField.
        ~BareField() = default;

        // Initialize the field, if it was constructed from the default constructor.
        // This should NOT be called if the field was constructed by providing
        // a FieldLayout.
        void initialize(Layout_t &);

        typedef std::deque<LField_t> container_t;

        LField_t& operator()(size_t i) {
            return lfields_m[i];
        }

        const LField_t& operator()(size_t i) const {
            return lfields_m[i];
        }


        const LField_t& operator[](size_t i) const {
            return lfields_m[i];
        }


        // Access to the layout.
        Layout_t &getLayout() const
        {
            PAssert(Layout != 0);
            return *Layout;
        }


        const Index& getIndex(unsigned d) const {return getLayout().getDomain()[d];}
        const NDIndex<Dim>& getDomain() const { return getLayout().getDomain(); }

        // Assignment from a constant.
        BareField<T, Dim>& operator=(T x);

        template <typename E>
        BareField<T, Dim>& operator=(const FieldExpression<E>& expr);

        void write(std::ostream& = std::cout);


    protected:
        container_t lfields_m;

    private:
        // Setup allocates all the LFields.  The various ctors call this.
        void setup();

        // How the local arrays are laid out.
        Layout_t *Layout;
    };
}

//////////////////////////////////////////////////////////////////////

#include "Field/BareField.hpp"

#endif
