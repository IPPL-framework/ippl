// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 * 
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef KOKKOS_BARE_FIELD_H
#define KOKKOS_BARE_FIELD_H

/***************************************************************************
 *
 * This is the user visible BareField of type T.
 * It doesn't even really do expression evaluation; that is
 * handled with the templates in Expressions.h
 *
 ***************************************************************************/

// include files
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
    template<class T,  unsigned Dim>
    class BareField : public FieldExpression< BareField<T, Dim> >
    {

    public:
        // Some externally visible typedefs and enums
        typedef T T_t;
        typedef FieldLayout<Dim> Layout_t;
        typedef LField<T,Dim> LField_t;
        enum { Dim_u = Dim };

        public:
        // A default constructor, which should be used only if the user calls the
        // 'initialize' function before doing anything else.  There are no special
        // checks in the rest of the BareField methods to check that the field has
        // been properly initialized.
        BareField();

        // Create a new BareField with a given layout and optional guard cells.
        BareField(Layout_t &);


        BareField(const BareField&) = default;

        // Destroy the BareField.
        ~BareField();

        // Initialize the field, if it was constructed from the default constructor.
        // This should NOT be called if the field was constructed by providing
        // a FieldLayout.
        void initialize(Layout_t &);

        typedef std::deque<LField_t> container_t;
        //   typedef typename container_t::iterator iterator;
        //   typedef typename container_t::const_iterator const_iterator;

        //   iterator begin() { return lfields_m.begin(); }
        //   iterator end() { return lfields_m.end(); }

        //   const_iterator begin() const { return lfields_m.begin(); }
        //   const_iterator end() const { return lfields_m.end(); }

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
        BareField<T,Dim>& operator=(T x)
        {
            for (auto& lf : lfields_m) {
                lf = x;
            }
            return *this;
        }

        // Assign another array.
        BareField<T,Dim>&
        operator=(const BareField<T,Dim>& rhs)
        {
            for (size_t i = 0; i < lfields_m.size(); ++i) {
                lfields_m[i] = rhs(i);
            }
            return *this;
        }


        template <typename E>
        inline BareField<T,Dim>& operator=(const FieldExpression<E>& expr) {
            for (size_t i = 0; i < lfields_m.size(); ++i) {
                lfields_m[i] = expr[i];
            }
            return *this;
        }

        void write(std::ostream& = std::cout);


    protected:
        container_t lfields_m;

    private:
        // Setup allocates all the LFields.  The various ctors call this.
        void setup();

        // How the local arrays are laid out.
        Layout_t *Layout;

        // robust method.  The externally visible get_single
        // calls this when it determines it needs it.
        void getsingle_bc(const NDIndex<Dim>&, T&) const;
    };

    //////////////////////////////////////////////////////////////////////

    //
    // Construct a BareField from nothing ... default case.
    //

    template< class T, unsigned Dim >
    inline
    BareField<T,Dim>::
    BareField()
    : Layout(0)			 // No layout yet.
    { }


    //
    // Construct a BareField from a FieldLayout.
    //

    template< class T, unsigned Dim >
    inline
    BareField<T,Dim>::
    BareField(Layout_t & l)
    : Layout(&l)			 // Just record the layout.
    {
    setup();			// Do the common setup chores.
    }


    //////////////////////////////////////////////////////////////////////

    template< class T, unsigned Dim >
    inline
    std::ostream& operator<<(std::ostream& out, const BareField<T,Dim>& a)
    {



    BareField<T,Dim>& nca = const_cast<BareField<T,Dim>&>(a);
    nca.write(out);
    return out;
    }
}

//////////////////////////////////////////////////////////////////////

#include "Field/BareField.hpp"

#endif // BARE_FIELD_H
