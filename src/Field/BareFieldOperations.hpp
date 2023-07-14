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
    template <typename BareField>
    typename BareField::value_type innerProduct(const BareField& f1, const BareField& f2) {
        using T                = typename BareField::value_type;
        constexpr unsigned Dim = BareField::dim;

        T sum                  = 0;
        auto view1             = f1.getView();
        auto view2             = f2.getView();
        using exec_space       = typename BareField::execution_space;
        using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;
        ippl::parallel_reduce(
            "Field::innerProduct(Field&, Field&)", f1.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const index_array_type& args, T& val) {
                val += apply(view1, args) * apply(view2, args);
            },
            Kokkos::Sum<T>(sum));
        if (f1.getLayout().isAllSerial()) {
            return sum;
        }
        T globalSum       = 0;
        MPI_Datatype type = get_mpi_datatype<T>(sum);
        MPI_Allreduce(&sum, &globalSum, 1, type, MPI_SUM, Comm->getCommunicator());
        return globalSum;
    }

    /*!
     * Computes the Lp-norm of a field
     * @param field field
     * @param p desired norm (default 2)
     * @return The desired norm of the field
     */
    template <typename BareField>
    typename BareField::value_type norm(const BareField& field, int p = 2) {
        using T                = typename BareField::value_type;
        constexpr unsigned Dim = BareField::dim;

        T local                = 0;
        auto view              = field.getView();
        using exec_space       = typename BareField::execution_space;
        using index_array_type = typename RangePolicy<Dim, exec_space>::index_array_type;
        switch (p) {
            case 0: {
                ippl::parallel_reduce(
                    "Field::norm(0)", field.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const index_array_type& args, T& val) {
                        T myVal = std::abs(apply(view, args));
                        if (myVal > val)
                            val = myVal;
                    },
                    Kokkos::Max<T>(local));
                if (field.getLayout().isAllSerial()) {
                    return local;
                }
                T globalMax       = 0;
                MPI_Datatype type = get_mpi_datatype<T>(local);
                MPI_Allreduce(&local, &globalMax, 1, type, MPI_MAX, Comm->getCommunicator());
                return globalMax;
            }
            default: {
                ippl::parallel_reduce(
                    "Field::norm(int) general", field.getFieldRangePolicy(),
                    KOKKOS_LAMBDA(const index_array_type& args, T& val) {
                        val += std::pow(std::abs(apply(view, args)), p);
                    },
                    Kokkos::Sum<T>(local));
                if (field.getLayout().isAllSerial()) {
                    return local;
                }
                T globalSum       = 0;
                MPI_Datatype type = get_mpi_datatype<T>(local);
                MPI_Allreduce(&local, &globalSum, 1, type, MPI_SUM, Comm->getCommunicator());
                return std::pow(globalSum, 1.0 / p);
            }
        }
    }
}  // namespace ippl
