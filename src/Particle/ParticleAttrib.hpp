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
//   particles (just as BareField is the primary element there).  This file
//   defines the necessary templated classes and functions to make
//   ParticleAttrib a capable expression-template participant.
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
#include "Ippl.h"

#include "Communicate/DataTypes.h"

#include "Utility/IpplTimings.h"

namespace ippl {

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::create(size_type n) {
        size_type required = *(this->localNum_mp) + n;
        if (this->size() < required) {
            int overalloc = Ippl::Comm->getDefaultOverallocation();
            this->realloc(required * overalloc);
        }
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::destroy(const Kokkos::View<int*>& deleteIndex,
                                                   const Kokkos::View<int*>& keepIndex,
                                                   size_type invalidCount) {
        // Replace all invalid particles in the valid region with valid
        // particles in the invalid region
        Kokkos::parallel_for(
            "ParticleAttrib::destroy()", invalidCount, KOKKOS_CLASS_LAMBDA(const size_t i) {
                dview_m(deleteIndex(i)) = dview_m(keepIndex(i));
            });
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::pack(void* buffer,
                                                const Kokkos::View<int*>& hash) const {
        using this_type     = ParticleAttrib<T, Properties...>;
        this_type* buffer_p = static_cast<this_type*>(buffer);
        auto& view          = buffer_p->dview_m;
        auto size           = hash.extent(0);
        if (view.extent(0) < size) {
            int overalloc = Ippl::Comm->getDefaultOverallocation();
            Kokkos::realloc(view, size * overalloc);
        }

        Kokkos::parallel_for(
            "ParticleAttrib::pack()", size,
            KOKKOS_CLASS_LAMBDA(const size_t i) { view(i) = dview_m(hash(i)); });
        Kokkos::fence();
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::unpack(void* buffer, size_type nrecvs) {
        using this_type     = ParticleAttrib<T, Properties...>;
        this_type* buffer_p = static_cast<this_type*>(buffer);
        auto& view          = buffer_p->dview_m;
        auto size           = dview_m.extent(0);
        size_type required  = *(this->localNum_mp) + nrecvs;
        if (size < required) {
            int overalloc = Ippl::Comm->getDefaultOverallocation();
            this->resize(required * overalloc);
        }

        size_type count = *(this->localNum_mp);
        Kokkos::parallel_for(
            "ParticleAttrib::unpack()", nrecvs,
            KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(count + i) = view(i); });
        Kokkos::fence();
    }

    template <typename T, class... Properties>
    // KOKKOS_INLINE_FUNCTION
    ParticleAttrib<T, Properties...>& ParticleAttrib<T, Properties...>::operator=(T x) {
        Kokkos::parallel_for(
            "ParticleAttrib::operator=()", *(this->localNum_mp),
            KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(i) = x; });
        return *this;
    }

    template <typename T, class... Properties>
    template <typename E, size_t N>
    // KOKKOS_INLINE_FUNCTION
    ParticleAttrib<T, Properties...>& ParticleAttrib<T, Properties...>::operator=(
        detail::Expression<E, N> const& expr) {
        using capture_type = detail::CapturedExpression<E, N>;
        capture_type expr_ = reinterpret_cast<const capture_type&>(expr);

        Kokkos::parallel_for(
            "ParticleAttrib::operator=()", *(this->localNum_mp),
            KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(i) = expr_(i); });
        return *this;
    }

    template <typename T, class... Properties>
    template <unsigned Dim, class M, class C, class PT>
    void ParticleAttrib<T, Properties...>::scatter(
        Field<T, Dim, M, C>& f, const ParticleAttrib<Vector<PT, Dim>, Properties...>& pp) const {
        static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");
        IpplTimings::startTimer(scatterTimer);
        using view_type = typename Field<T, Dim, M, C>::view_type;
        view_type view  = f.getView();

        const M& mesh = f.get_mesh();

        using vector_type = typename M::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;

        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const int nghost               = f.getNghost();

        Kokkos::parallel_for(
            "ParticleAttrib::scatter", *(this->localNum_mp), KOKKOS_CLASS_LAMBDA(const size_t idx) {
                // find nearest grid point
                vector_type l           = (pp(idx) - origin) * invdx + 0.5;
                Vector<int, Dim> index  = l;
                vector_type whi         = l - index;
                vector_type wlo         = 1.0 - whi;

                Vector<size_t, Dim> args = index - lDom.first() + nghost;

                // scatter
                const value_type& val = dview_m(idx);
                detail::scatterToField(std::make_index_sequence<1 << Dim>{}, view, wlo, whi, args,
                                       val);
            });
        IpplTimings::stopTimer(scatterTimer);

        static IpplTimings::TimerRef accumulateHaloTimer = IpplTimings::getTimer("accumulateHalo");
        IpplTimings::startTimer(accumulateHaloTimer);
        f.accumulateHalo();
        IpplTimings::stopTimer(accumulateHaloTimer);
    }

    template <typename T, class... Properties>
    template <unsigned Dim, class M, class C, typename P2>
    void ParticleAttrib<T, Properties...>::gather(
        Field<T, Dim, M, C>& f, const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp) {
        static IpplTimings::TimerRef fillHaloTimer = IpplTimings::getTimer("fillHalo");
        IpplTimings::startTimer(fillHaloTimer);
        f.fillHalo();
        IpplTimings::stopTimer(fillHaloTimer);

        static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather");
        IpplTimings::startTimer(gatherTimer);
        const typename Field<T, Dim, M, C>::view_type view = f.getView();

        const M& mesh = f.get_mesh();

        using vector_type = typename M::vector_type;

        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const int nghost               = f.getNghost();

        Kokkos::parallel_for(
            "ParticleAttrib::gather", *(this->localNum_mp), KOKKOS_CLASS_LAMBDA(const size_t idx) {
                // find nearest grid point
                vector_type l           = (pp(idx) - origin) * invdx + 0.5;
                Vector<int, Dim> index  = l;
                vector_type whi		= l - index;
                vector_type wlo		= 1.0 - whi;

                Vector<size_t, Dim> args = index - lDom.first() + nghost;

                // gather
                dview_m(idx) = detail::gatherFromField(std::make_index_sequence<1 << Dim>{}, view,
                                                       wlo, whi, args);
            });
        IpplTimings::stopTimer(gatherTimer);
    }

    /*
     * Non-class function
     *
     */

    template <typename P1, unsigned Dim, class M, class C, typename P2, class... Properties>
    inline void scatter(const ParticleAttrib<P1, Properties...>& attrib, Field<P1, Dim, M, C>& f,
                        const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp) {
        attrib.scatter(f, pp);
    }

    template <typename P1, unsigned Dim, class M, class C, typename P2, class... Properties>
    inline void gather(ParticleAttrib<P1, Properties...>& attrib, Field<P1, Dim, M, C>& f,
                       const ParticleAttrib<Vector<P2, Dim>, Properties...>& pp) {
        attrib.gather(f, pp);
    }

#define DefineParticleReduction(fun, name, op, MPI_Op)                       \
    template <typename T, class... Properties>                               \
    T ParticleAttrib<T, Properties...>::name() {                             \
        T temp = 0.0;                                                        \
        Kokkos::parallel_reduce(                                             \
            "fun", *(this->localNum_mp),                                     \
            KOKKOS_CLASS_LAMBDA(const size_t i, T& valL) {                   \
                T myVal = dview_m(i);                                        \
                op;                                                          \
            },                                                               \
            Kokkos::fun<T>(temp));                                           \
        T globaltemp      = 0.0;                                             \
        MPI_Datatype type = get_mpi_datatype<T>(temp);                       \
        MPI_Allreduce(&temp, &globaltemp, 1, type, MPI_Op, Ippl::getComm()); \
        return globaltemp;                                                   \
    }

    DefineParticleReduction(Sum, sum, valL += myVal, MPI_SUM)
    DefineParticleReduction(Max, max, if (myVal > valL) valL = myVal, MPI_MAX)
    DefineParticleReduction(Min, min, if (myVal < valL) valL = myVal, MPI_MIN)
    DefineParticleReduction(Prod, prod, valL *= myVal, MPI_PROD)
}  // namespace ippl
