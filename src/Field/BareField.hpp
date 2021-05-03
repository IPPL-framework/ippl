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
#include "Utility/Inform.h"
#include "Utility/IpplInfo.h"
#include "Ippl.h"
#include "Communicate/DataTypes.h"

#include <map>
#include <utility>
#include <cstdlib>

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim>
        struct isExpression<BareField<T, Dim>> : std::true_type {};
    }

    template <typename T, unsigned Dim>
    BareField<T, Dim>::BareField()
    : nghost_m(1)
    , layout_m(nullptr)
    { }


    template <typename T, unsigned Dim>
    BareField<T, Dim>::BareField(Layout_t& l, int nghost)
    : nghost_m(nghost)
//     , owned_m(0)
    , layout_m(&l)
    {
        setup();
    }

    template <typename T, unsigned Dim>
    void BareField<T, Dim>::initialize(Layout_t& l, int nghost) {
        if (layout_m == 0) {
            layout_m = &l;
            nghost_m = nghost;
            setup();
        }
    }


    template <typename T, unsigned Dim>
    void BareField<T, Dim>::setup() {
        static_assert(Dim == 2 || Dim == 3, "Only 2D and 3D fields supported at the momment!");

        owned_m = layout_m->getLocalNDIndex();

        if constexpr(Dim == 2) {
            this->resize(owned_m[0].length() + 2 * nghost_m,
                         owned_m[1].length() + 2 * nghost_m);
        } else if constexpr(Dim == 3) {
            this->resize(owned_m[0].length() + 2 * nghost_m,
                         owned_m[1].length() + 2 * nghost_m,
                         owned_m[2].length() + 2 * nghost_m);
        }
    }


    template <typename T, unsigned Dim>
    template <typename ...Args>
    void BareField<T, Dim>::resize(Args... args) {
        Kokkos::resize(dview_m, args...);
    }


    template <typename T, unsigned Dim>
    void BareField<T, Dim>::fillHalo() {
        if(Ippl::Comm->size() > 1) {
            halo_m.fillHalo(dview_m, layout_m, nghost_m);
        }
    }


    template <typename T, unsigned Dim>
    void BareField<T, Dim>::accumulateHalo() {
        if(Ippl::Comm->size() > 1) {
            halo_m.accumulateHalo(dview_m, layout_m, nghost_m);
        }
    }



    template <typename T, unsigned Dim>
    template <unsigned dim, std::enable_if_t<(dim == 2), bool>>
    BareField<T, Dim>& BareField<T, Dim>::operator=(T x) {
        policy_type policy = getRangePolicy(nghost_m);

        Kokkos::parallel_for("BareField::operator=(T)",
                            policy,
                            KOKKOS_CLASS_LAMBDA(const size_t i,
                                                const size_t j)
                            {
                                dview_m(i, j) = x;
                            });
        return *this;
    }


    template <typename T, unsigned Dim>
    template <unsigned dim, std::enable_if_t<(dim == 3), bool>>
    BareField<T, Dim>& BareField<T, Dim>::operator=(T x) {
        policy_type policy = getRangePolicy(nghost_m);

        Kokkos::parallel_for("BareField::operator=(T)",
                            policy,
                            KOKKOS_CLASS_LAMBDA(const size_t i,
                                                const size_t j,
                                                const size_t k)
                            {
                                dview_m(i, j, k) = x;
                            });
        return *this;
    }


    template <typename T, unsigned Dim>
    template <typename E, size_t N, unsigned dim, std::enable_if_t<(dim == 2), bool>>
    BareField<T, Dim>& BareField<T, Dim>::operator=(const detail::Expression<E, N>& expr) {
        using capture_type = detail::CapturedExpression<E, N>;
        capture_type expr_ = reinterpret_cast<const capture_type&>(expr);

        policy_type policy = getRangePolicy(nghost_m);
        Kokkos::parallel_for("BareField::operator=(const Expression&)",
                             policy,
                             KOKKOS_CLASS_LAMBDA(const size_t i,
                                                 const size_t j)
                             {
                                dview_m(i, j) = expr_(i, j);
                             });
        return *this;
    }


    template <typename T, unsigned Dim>
    template <typename E, size_t N, unsigned dim, std::enable_if_t<(dim == 3), bool>>
    BareField<T, Dim>& BareField<T, Dim>::operator=(const detail::Expression<E, N>& expr) {
        using capture_type = detail::CapturedExpression<E, N>;
        capture_type expr_ = reinterpret_cast<const capture_type&>(expr);

        policy_type policy = getRangePolicy(nghost_m);
        Kokkos::parallel_for("BareField::operator=(const Expression&)",
                             policy,
                             KOKKOS_CLASS_LAMBDA(const size_t i,
                                                 const size_t j,
                                                 const size_t k)
                             {
                                dview_m(i, j, k) = expr_(i, j, k);
                             });
        return *this;
    }


    template <typename T, unsigned Dim>
    void BareField<T,Dim>::write(std::ostream& out) const {
        Kokkos::fence();
        detail::write<T>(dview_m, out);
    }


    #define DefineReduction(fun, name, op, MPI_Op)                              \
    template <typename T, unsigned Dim>                                         \
    template <unsigned dim, std::enable_if_t<(dim == 2), bool>>                 \
    T BareField<T, Dim>::name(int nghost) {                                     \
        T temp = 0.0;                                                           \
        policy_type policy = getRangePolicy(nghost);                            \
        Kokkos::parallel_reduce("fun",                                          \
                                policy,                                         \
                                KOKKOS_CLASS_LAMBDA(const size_t i,             \
                                                    const size_t j,             \
                                                    T& valL)                    \
                                {                                               \
                                    T myVal = dview_m(i, j);                    \
                                    op;                                         \
                                }, Kokkos::fun<T>(temp));                       \
        T globaltemp = 0.0;                                                     \
        MPI_Datatype type = get_mpi_datatype<T>(temp);                          \
        MPI_Allreduce(&temp, &globaltemp, 1, type, MPI_Op, Ippl::getComm());    \
        return globaltemp;                                                      \
    }                                                                           \
                                                                                \
                                                                                \
    template <typename T, unsigned Dim>                                         \
    template <unsigned dim, std::enable_if_t<(dim == 3), bool>>                 \
    T BareField<T, Dim>::name(int nghost) {                                     \
        T temp = 0.0;                                                           \
        policy_type policy = getRangePolicy(nghost);                            \
        Kokkos::parallel_reduce("fun",                                          \
                                policy,                                         \
                                KOKKOS_CLASS_LAMBDA(const size_t i,             \
                                                    const size_t j,             \
                                                    const size_t k,             \
                                                    T& valL)                    \
                                {                                               \
                                    T myVal = dview_m(i, j, k);                 \
                                    op;                                         \
                                }, Kokkos::fun<T>(temp));                       \
        T globaltemp = 0.0;                                                     \
        MPI_Datatype type = get_mpi_datatype<T>(temp);                          \
        MPI_Allreduce(&temp, &globaltemp, 1, type, MPI_Op, Ippl::getComm());    \
        return globaltemp;                                                      \
    }

    DefineReduction(Sum,  sum,  valL += myVal, MPI_SUM)
    DefineReduction(Max,  max,  if(myVal > valL) valL = myVal, MPI_MAX)
    DefineReduction(Min,  min,  if(myVal < valL) valL = myVal, MPI_MIN)
    DefineReduction(Prod, prod, valL *= myVal, MPI_PROD)
}
