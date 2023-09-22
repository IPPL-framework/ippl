//
// Class ParticleAttrib
//   Templated class for all particle attribute classes.
//
//   This templated class is used to represent a single particle attribute.
//   An attribute is one data element within a particle object, and is
//   stored as a Kokkos::View. This class stores the type information for the
//   attribute, and provides methods to create and destroy new items, and
//   to perform operations involving this attribute with others.
//
//   ParticleAttrib is the primary element involved in expressions for
//   particles (just as LField is the primary element there).  This file
//   defines the necessary templated classes and functions to make
//   ParticleAttrib a capable expression-template participant.
//
#ifndef IPPL_PARTICLE_ATTRIB_H
#define IPPL_PARTICLE_ATTRIB_H

#include "Expression/IpplExpressions.h"

#include "Interpolation/CIC.h"
#include "Particle/ParticleAttribBase.h"

namespace ippl {

    // ParticleAttrib class definition
    template <typename T, class... Properties>
    class ParticleAttrib : public detail::ParticleAttribBase<>::with_properties<Properties...>,
                           public detail::Expression<
                               ParticleAttrib<T, Properties...>,
                               sizeof(typename detail::ViewType<T, 1, Properties...>::view_type)> {
    public:
        typedef T value_type;
        constexpr static unsigned dim = 1;

        using Base = typename detail::ParticleAttribBase<>::with_properties<Properties...>;

        using hash_type = typename Base::hash_type;

        using view_type  = typename detail::ViewType<T, 1, Properties...>::view_type;

        using HostMirror = typename view_type::host_mirror_type;

        using memory_space    = typename view_type::memory_space;
        using execution_space = typename view_type::execution_space;

        using size_type = detail::size_type;

        // Create storage for M particle attributes.  The storage is uninitialized.
        // New items are appended to the end of the array.
        void create(size_type) override;

        /*!
         * Particle deletion function. Partition the particles into a valid region
         * and an invalid region.
         * @param deleteIndex List of indices of invalid particles in the valid region
         * @param keepIndex List of indices of valid particles in the invalid region
         * @param invalidCount Number of invalid particles in the valid region
         */
        void destroy(const hash_type& deleteIndex, const hash_type& keepIndex,
                     size_type invalidCount) override;

        void pack(const hash_type&) override;

        void unpack(size_type) override;

        void serialize(detail::Archive<memory_space>& ar, size_type nsends) override {
            ar.serialize(buf_m, nsends);
        }

        void deserialize(detail::Archive<memory_space>& ar, size_type nrecvs) override {
            ar.deserialize(buf_m, nrecvs);
        }

        virtual ~ParticleAttrib() = default;

        size_type size() const override { return dview_m.extent(0); }

        size_type packedSize(const size_type count) const override {
            return count * sizeof(value_type);
        }

        void resize(size_type n) { Kokkos::resize(dview_m, n); }

        void realloc(size_type n) { Kokkos::realloc(dview_m, n); }

        void print() {
            HostMirror hview = Kokkos::create_mirror_view(dview_m);
            Kokkos::deep_copy(hview, dview_m);
            for (size_type i = 0; i < *(this->localNum_mp); ++i) {
                std::cout << hview(i) << std::endl;
            }
        }

        KOKKOS_INLINE_FUNCTION T& operator()(const size_t i) const { return dview_m(i); }

        view_type& getView() { return dview_m; }

        const view_type& getView() const { return dview_m; }

        HostMirror getHostMirror() { return Kokkos::create_mirror(dview_m); }

        /*!
         * Assign the same value to the whole attribute.
         */
        // KOKKOS_INLINE_FUNCTION
        ParticleAttrib<T, Properties...>& operator=(T x);

        /*!
         * Assign an arbitrary particle attribute expression
         * @tparam E expression type
         * @tparam N size of the expression, this is necessary for running on the
         * device since otherwise it does not allocate enough memory
         * @param expr is the expression
         */
        template <typename E, size_t N>
        // KOKKOS_INLINE_FUNCTION
        ParticleAttrib<T, Properties...>& operator=(detail::Expression<E, N> const& expr);

        //     // scatter the data from this attribute onto the given Field, using
        //     // the given Position attribute
        template <typename Field, typename P2>
        void scatter(Field& f,
                     const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp) const;

        template <typename Field, typename P2>
        void gather(Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp);

        T sum();
        T max();
        T min();
        T prod();

    private:
        view_type dview_m;
        view_type buf_m;
    };
}  // namespace ippl

#include "Particle/ParticleAttrib.hpp"

#endif
