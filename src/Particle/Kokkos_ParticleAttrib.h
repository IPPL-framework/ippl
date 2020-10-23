// -*- C++ -*-
/***************************************************************************
 *
 * The IPPL Framework
 *
 *
 * Visit http://people.web.psi.ch/adelmann/ for more details
 *
 ***************************************************************************/

#ifndef IPPL_PARTICLE_ATTRIB_H
#define IPPL_PARTICLE_ATTRIB_H

/*
 * ParticleAttrib - Templated class for all particle attribute classes.
 *
 * This templated class is used to represent a single particle attribute.
 * An attribute is one data element within a particle object, and is
 * stored as an array.  This class stores the type information for the
 * attribute, and provides methods to create and destroy new items, and
 * to perform operations involving this attribute with others.  It also
 * provides iterators to allow the user to operate on single particles
 * instead of the entire array.
 *
 * ParticleAttrib is the primary element involved in expressions for
 * particles (just as Field is the primary element there).  This file
 * defines the necessary templated classes and functions to make
 * ParticleAttrib a capable expression-template participant.
 *
 * For some types such as Vektor, Tenzor, etc. which have multiple items,
 * we want to involve just the Nth item from each data element in an
 * expression.  The () operator here returns an object of type
 * ParticleAttribElem, which will use the () operator on each individual
 * element to access an item over which one can iterate and get just the
 * Nth item from each data element.  For example, if we have an attribute
 * like this:
 *             ParticleAttrib< Vektor<float, 4> > Data
 * we can involve just the 2nd item of each Vektor in an expression by
 * referring to this as
 *             Data(1)
 * which returns an object of type ParticleAttribElem that knows to return
 * just the 2nd item from each Vektor.  ParticleAttribElem is also expression-
 * template-aware; in fact, it is intended primarily for use in expressions
 * and not in many other situations.  The ParticleAttribElem will use the
 * () operator to get the Nth item out of each data element, so this requires
 * the user to define operator () for the particle attribute type being
 * used (for Vektor, Tenzor, etc., this has already been done).  This same
 * thing has been done for operator () involving either one or two indices,
 * which is needed to get the i,j element of a Tenzor, for example.
 *
 * To perform gather/scatter type operations involving sparse indices, in
 * which the sparse indices represent a list of points in a dense field
 * onto which we want to gather/scatter values, you can use the [] operator
 * to get a SubParticleAttrib object that knows about the particle
 * elements and associated sparse index points.  This allows us to have
 * the syntax
 *    P[S] = expr(A[S])
 * where P is a ParticleAttrib, A is some other object such as a Field that
 * can be indexed by an SIndex, and S is an SIndex object.  In this case,
 * the length of the ParticleAttrib would be changed to match the number
 * of local points in the SIndex, and the expression would be evaluated at
 * all the points in the SIndex and stored into P.  It also allows the
 * syntax
 *    A[S] = expr(B[S], P[S])
 * where A, B are things like Field, S in an SIndex, and P is a ParticleAttrib.
 * Here, the LHS is assigned, at all the points in the SIndex, to the values
 * of the expression, which can include a ParticleAttrib only if it is
 * indexed by an SIndex.  This is because SubParticleAttrib contains the
 * ability to provide an iterator with the right interface for the expression
 * evaluation.
 */


#include "Types/ViewTypes.h"
#include "Expression/IpplExpressions.h"

namespace ippl {

    template<class... Properties>
    class ParticleAttribBase {

    public:
        typedef typename detail::ViewType<bool, 1, Properties...>::view_type boolean_view_type;

        virtual void create(size_t) = 0;

        virtual void destroy(boolean_view_type, Kokkos::View<int*>, size_t) = 0;

        virtual ~ParticleAttribBase() = default;

    };

    // ParticleAttrib class definition
    template <typename T, class... Properties>
    class ParticleAttrib : public ParticleAttribBase<Properties...>
                         , public detail::ViewType<T, 1, Properties...>::view_type
                         , public Expression<ParticleAttrib<T, Properties...>,
                                             sizeof(typename detail::ViewType<T, 1, Properties...>::view_type)>
    {
    public:
        typedef T value_type;
        using boolean_view_type = typename ParticleAttribBase<Properties...>::boolean_view_type;
        using view_type = typename detail::ViewType<T, 1, Properties...>::view_type;

        // Create storage for M particle attributes.  The storage is uninitialized.
        // New items are appended to the end of the array.
        virtual void create(size_t);

        virtual void destroy(boolean_view_type, Kokkos::View<int*> cc, size_t);


        virtual ~ParticleAttrib() = default;
       
        size_t size() const {
            return this->extent(0);
        }

        void resize(size_t n) {
            Kokkos::resize(*this, n);
        }

        void print() {
            typename view_type::HostMirror hview = Kokkos::create_mirror_view(*this);
            Kokkos::deep_copy(hview, *this);
            for (size_t i = 0; i < this->size(); ++i) {
                std::cout << hview(i) << std::endl;
            }
        }


        /*!
         * Assign the same value to the whole attribute.
         */
        ParticleAttrib<T, Properties...>& operator=(T x);

        /*!
         * Assign an arbitrary particle attribute expression
         * @tparam E expression type
         * @tparam N size of the expression, this is necessary for running on the
         * device since otherwise it does not allocate enough memory
         * @param expr is the expression
         */
        template <typename E, size_t N>
        ParticleAttrib<T, Properties...>& operator=(Expression<E, N> const& expr);


        //     // scatter the data from this attribute onto the given Field, using
//     // the given Position attribute
        template <unsigned Dim, class M, class C, typename P2>
        void
        scatter(Field<T, Dim, M, C>& f,
                const ParticleAttrib<Vector<P2, Dim>, Properties... >& pp) const;


        template <unsigned Dim, class M, class C, typename P2>
        void
        gather(const Field<T, Dim, M, C>& f,
               const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp);
    };
}

#include "Particle/Kokkos_ParticleAttrib.hpp"

#endif