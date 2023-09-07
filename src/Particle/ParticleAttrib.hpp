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
            int overalloc = Comm->getDefaultOverallocation();
            this->realloc(required * overalloc);
        }
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::destroy(const hash_type& deleteIndex,
                                                   const hash_type& keepIndex,
                                                   size_type invalidCount) {
        // Replace all invalid particles in the valid region with valid
        // particles in the invalid region
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::destroy()", policy_type(0, invalidCount),
            KOKKOS_CLASS_LAMBDA(const size_t i) {
                dview_m(deleteIndex(i)) = dview_m(keepIndex(i));
            });
    }

    template <typename T, class... Properties>
    void ParticleAttrib<T, Properties...>::pack(void* buffer, const hash_type& hash) const {
        using this_type     = ParticleAttrib<T, Properties...>;
        this_type* buffer_p = static_cast<this_type*>(buffer);
        auto& view          = buffer_p->dview_m;
        auto size           = hash.extent(0);
        if (view.extent(0) < size) {
            int overalloc = Comm->getDefaultOverallocation();
            Kokkos::realloc(view, size * overalloc);
        }

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::pack()", policy_type(0, size),
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
            int overalloc = Comm->getDefaultOverallocation();
            this->resize(required * overalloc);
        }

        size_type count   = *(this->localNum_mp);
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::unpack()", policy_type(0, nrecvs),
            KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(count + i) = view(i); });
        Kokkos::fence();
    }

    template <typename T, class... Properties>
    // KOKKOS_INLINE_FUNCTION
    ParticleAttrib<T, Properties...>& ParticleAttrib<T, Properties...>::operator=(T x) {
        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::operator=()", policy_type(0, *(this->localNum_mp)),
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

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::operator=()", policy_type(0, *(this->localNum_mp)),
            KOKKOS_CLASS_LAMBDA(const size_t i) { dview_m(i) = expr_(i); });
        return *this;
    }

    template <typename T, class... Properties>
    template <typename Field, class PT>
    void ParticleAttrib<T, Properties...>::scatter(
        Field& f, const ParticleAttrib<Vector<PT, Field::dim>, Properties...>& pp) const {
        constexpr unsigned Dim = Field::dim;
        using PositionType     = typename Field::Mesh_t::value_type;

        static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");
        IpplTimings::startTimer(scatterTimer);
        using view_type = typename Field::view_type;
        view_type view  = f.getView();

        using mesh_type       = typename Field::Mesh_t;
        const mesh_type& mesh = f.get_mesh();

        using vector_type = typename mesh_type::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;

        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const int nghost               = f.getNghost();

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::scatter", policy_type(0, *(this->localNum_mp)),
            KOKKOS_CLASS_LAMBDA(const size_t idx) {
                // find nearest grid point
                vector_type l                        = (pp(idx) - origin) * invdx + 0.5;
                Vector<int, Field::dim> index        = l;
                Vector<PositionType, Field::dim> whi = l - index;
                Vector<PositionType, Field::dim> wlo = 1.0 - whi;

                Vector<size_t, Field::dim> args = index - lDom.first() + nghost;

                // scatter
                const value_type& val = dview_m(idx);
                detail::scatterToField(std::make_index_sequence<1 << Field::dim>{}, view, wlo, whi,
                                       args, val);
            });
        IpplTimings::stopTimer(scatterTimer);

        static IpplTimings::TimerRef accumulateHaloTimer = IpplTimings::getTimer("accumulateHalo");
        IpplTimings::startTimer(accumulateHaloTimer);
        f.accumulateHalo();
        IpplTimings::stopTimer(accumulateHaloTimer);
    }

    template <typename T, class... Properties>
    template <typename Field, typename P2>
    void ParticleAttrib<T, Properties...>::gather(
        Field& f, const ParticleAttrib<Vector<P2, Field::dim>, Properties...>& pp) {
        constexpr unsigned Dim = Field::dim;
        using PositionType     = typename Field::Mesh_t::value_type;

        static IpplTimings::TimerRef fillHaloTimer = IpplTimings::getTimer("fillHalo");
        IpplTimings::startTimer(fillHaloTimer);
        f.fillHalo();
        IpplTimings::stopTimer(fillHaloTimer);

        static IpplTimings::TimerRef gatherTimer = IpplTimings::getTimer("gather");
        IpplTimings::startTimer(gatherTimer);
        const typename Field::view_type view = f.getView();

        using mesh_type       = typename Field::Mesh_t;
        const mesh_type& mesh = f.get_mesh();

        using vector_type = typename mesh_type::vector_type;

        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const int nghost               = f.getNghost();

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::gather", policy_type(0, *(this->localNum_mp)),
            KOKKOS_CLASS_LAMBDA(const size_t idx) {
                // find nearest grid point
                vector_type l                        = (pp(idx) - origin) * invdx + 0.5;
                Vector<int, Field::dim> index        = l;
                Vector<PositionType, Field::dim> whi = l - index;
                Vector<PositionType, Field::dim> wlo = 1.0 - whi;

                Vector<size_t, Field::dim> args = index - lDom.first() + nghost;

                // gather
                dview_m(idx) = detail::gatherFromField(std::make_index_sequence<1 << Field::dim>{},
                                                       view, wlo, whi, args);
            });
        IpplTimings::stopTimer(gatherTimer);
    }

    /*
     * Non-class function
     *
     */

    template <typename Attrib1, typename Field, typename Attrib2>
    inline void scatter(const Attrib1& attrib, Field& f, const Attrib2& pp) {
        attrib.scatter(f, pp);
    }

    template <typename Attrib1, typename Field, typename Attrib2>
    inline void gather(Attrib1& attrib, Field& f, const Attrib2& pp) {
        attrib.gather(f, pp);
    }

#define DefineParticleReduction(fun, name, op, MPI_Op)                               \
    template <typename T, class... Properties>                                       \
    T ParticleAttrib<T, Properties...>::name() {                                     \
        T temp            = 0.0;                                                     \
        using policy_type = Kokkos::RangePolicy<execution_space>;                    \
        Kokkos::parallel_reduce(                                                     \
            "fun", policy_type(0, *(this->localNum_mp)),                             \
            KOKKOS_CLASS_LAMBDA(const size_t i, T& valL) {                           \
                T myVal = dview_m(i);                                                \
                op;                                                                  \
            },                                                                       \
            Kokkos::fun<T>(temp));                                                   \
        T globaltemp      = 0.0;                                                     \
        MPI_Datatype type = get_mpi_datatype<T>(temp);                               \
        MPI_Allreduce(&temp, &globaltemp, 1, type, MPI_Op, Comm->getCommunicator()); \
        return globaltemp;                                                           \
    }

    DefineParticleReduction(Sum, sum, valL += myVal, MPI_SUM)
    DefineParticleReduction(Max, max, if (myVal > valL) valL = myVal, MPI_MAX)
    DefineParticleReduction(Min, min, if (myVal < valL) valL = myVal, MPI_MIN)
    DefineParticleReduction(Prod, prod, valL *= myVal, MPI_PROD)
}  // namespace ippl
