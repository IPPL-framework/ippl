//
// File BareFieldOperations
//   Norms and a scalar product for fields
//
// Copyright (c) 2023 Paul Scherrer Institut, Villigen PSI, Switzerland
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
    /*!
     * Computes the inner product of two fields
     * @param f1 first field
     * @param f2 second field
     * @return Result of f1^T f2
     */
    template <typename T, unsigned Dim>
    T innerProduct(const BareField<T, Dim>& f1, const BareField<T, Dim>& f2) {
        T sum      = 0;
        auto view1 = f1.getView();
        auto view2 = f2.getView();
        Kokkos::parallel_reduce(
            "Field::innerProduct(Field&, Field&)", f1.getRangePolicy(),
            detail::functorize<Dim, T>(KOKKOS_LAMBDA<typename... Idx>(const Idx... args, T& val) {
                val += view1(args...) * view2(args...);
            }),
            Kokkos::Sum<T>(sum));
        T globalSum       = 0;
        MPI_Datatype type = get_mpi_datatype<T>(sum);
        MPI_Allreduce(&sum, &globalSum, 1, type, MPI_SUM, Ippl::getComm());
        return globalSum;
    }

    /*!
     * Computes the Lp-norm of a field
     * @param field field
     * @param p desired norm (default 2)
     * @return The desired norm of the field
     */
    template <typename T, unsigned Dim>
    T norm(const BareField<T, Dim>& field, int p = 2) {
        T local   = 0;
        auto view = field.getView();
        switch (p) {
            case 0: {
                Kokkos::parallel_reduce("Field::norm(0)", field.getRangePolicy(),
                                        detail::functorize<Dim, T>(KOKKOS_LAMBDA<typename... Idx>(
                                            const Idx... args, T& val) {
                                            T myVal = std::abs(view(args...));
                                            if (myVal > val)
                                                val = myVal;
                                        }),
                                        Kokkos::Max<T>(local));
                T globalMax       = 0;
                MPI_Datatype type = get_mpi_datatype<T>(local);
                MPI_Allreduce(&local, &globalMax, 1, type, MPI_MAX, Ippl::getComm());
                return globalMax;
            }
            case 2:
                return std::sqrt(innerProduct(field, field));
            default: {
                Kokkos::parallel_reduce("Field::norm(int) general", field.getRangePolicy(),
                                        detail::functorize<Dim, T>(KOKKOS_LAMBDA<typename... Idx>(
                                            const Idx... args, T& val) {
                                            val += std::pow(std::abs(view(args...)), p);
                                        }),
                                        Kokkos::Sum<T>(local));
                T globalSum       = 0;
                MPI_Datatype type = get_mpi_datatype<T>(local);
                MPI_Allreduce(&local, &globalSum, 1, type, MPI_SUM, Ippl::getComm());
                return std::pow(globalSum, 1.0 / p);
            }
        }
    }
}  // namespace ippl
