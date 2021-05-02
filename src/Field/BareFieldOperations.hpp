//
// File BareFieldOperations
//   Norms, and a scalar product for fields
//
// Copyright (c) 2021 Paul Scherrer Institut, Villigen PSI, Switzerland
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

namespace ippl {
    #define DefineBinaryBareFieldOperation2D(fun, op, kokkos_op)        \
    template <typename T, unsigned Dim,                                 \
              std::enable_if_t<(Dim == 2), bool> = true>                \
    T fun(const BareField<T, Dim>& bf1,                                 \
          const BareField<T, Dim>& bf2,                                 \
          const int nghost = 0)                                         \
    {                                                                   \
        T lres = 0;                                                     \
        Kokkos::parallel_reduce("BareField::fun",                       \
                                bf1.getRangePolicy(nghost),             \
                                KOKKOS_LAMBDA(const size_t i,           \
                                              const size_t j,           \
                                              T& val)                   \
                                {                                       \
                                    op;                                 \
                                }, kokkos_op<T>(lres));                 \
        T gres = 0;                                                     \
        MPI_Datatype type = get_mpi_datatype<T>(lres);                  \
        MPI_Allreduce(&lres, &gres, 1, type, MPI_SUM, Ippl::getComm()); \
        return gres;                                                    \
    }


    #define DefineBinaryBareFieldOperation3D(fun, op, kokkos_op)        \
    template <typename T, unsigned Dim,                                 \
              std::enable_if_t<(Dim == 3), bool> = true>                \
    T fun(const BareField<T, Dim>& bf1,                                 \
          const BareField<T, Dim>& bf2,                                 \
          const int nghost = 0)                                         \
    {                                                                   \
        T lres = 0;                                                     \
        Kokkos::parallel_reduce("BareField::fun",                       \
                                bf1.getRangePolicy(nghost),             \
                                KOKKOS_LAMBDA(const size_t i,           \
                                              const size_t j,           \
                                              const size_t k,           \
                                              T& val)                   \
                                {                                       \
                                    op;                                 \
                                }, kokkos_op<T>(lres));                 \
        T gres = 0;                                                     \
        MPI_Datatype type = get_mpi_datatype<T>(lres);                  \
        MPI_Allreduce(&lres, &gres, 1, type, MPI_SUM, Ippl::getComm()); \
        return gres;                                                    \
    }


    #define DefineUnaryBareFieldOperation2D(fun, op, kokkos_op)         \
    template <typename T, unsigned Dim,                                 \
              std::enable_if_t<(Dim == 2), bool> = true>                \
    T fun(const BareField<T, Dim>& bf,                                  \
          const int nghost = 0)                                         \
    {                                                                   \
        T lres = 0;                                                     \
        Kokkos::parallel_reduce("BareField::fun",                       \
                                bf.getRangePolicy(nghost),              \
                                KOKKOS_LAMBDA(const size_t i,           \
                                              const size_t j,           \
                                              T& val)                   \
                                {                                       \
                                    op;                                 \
                                }, kokkos_op<T>(lres));                 \
        T gres = 0;                                                     \
        MPI_Datatype type = get_mpi_datatype<T>(lres);                  \
        MPI_Allreduce(&lres, &gres, 1, type, MPI_SUM, Ippl::getComm()); \
        return gres;                                                    \
    }


    #define DefineUnaryBareFieldOperation3D(fun, op, kokkos_op)         \
    template <typename T, unsigned Dim,                                 \
              std::enable_if_t<(Dim == 3), bool> = true>                \
    T fun(const BareField<T, Dim>& bf,                                  \
          const int nghost = 0)                                         \
    {                                                                   \
        T lres = 0;                                                     \
        Kokkos::parallel_reduce("BareField::fun",                       \
                                bf.getRangePolicy(nghost),              \
                                KOKKOS_LAMBDA(const size_t i,           \
                                              const size_t j,           \
                                              const size_t k,           \
                                              T& val)                   \
                                {                                       \
                                    op;                                 \
                                }, kokkos_op<T>(lres));                 \
        T gres = 0;                                                     \
        MPI_Datatype type = get_mpi_datatype<T>(lres);                  \
        MPI_Allreduce(&lres, &gres, 1, type, MPI_SUM, Ippl::getComm()); \
        return gres;                                                    \
    }


    DefineBinaryBareFieldOperation2D(innerProduct,
                                     val += bf1(i, j) * bf2(i, j);,
                                     Kokkos::Sum)
    DefineBinaryBareFieldOperation3D(innerProduct,
                                     val += bf1(i, j, k) * bf2(i, j, k);,
                                     Kokkos::Sum)

    DefineUnaryBareFieldOperation2D(norm1,
                                    val += std::abs(bf(i, j)),
                                    Kokkos::Sum)
    DefineUnaryBareFieldOperation3D(norm1,
                                    val += std::abs(bf(i, j, k)),
                                    Kokkos::Sum)

    DefineUnaryBareFieldOperation2D(normInf,
                                    val = std::abs(bf(i, j)),
                                    Kokkos::Max)
    DefineUnaryBareFieldOperation3D(normInf,
                                    val = std::abs(bf(i, j, k)),
                                    Kokkos::Max)

    template <typename T, unsigned Dim>
    T norm2(const BareField<T, Dim>& bf, const int nghost = 0) {
        return std::sqrt(innerProduct(bf, bf, nghost));
    }
}
