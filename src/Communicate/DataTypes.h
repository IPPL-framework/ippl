//
// File DataTypes
//   Definition of MPI types following the implementation of Boost.MPI.
//
// Copyright (c) 2017, Matthias Frey, Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// Implemented as part of the PhD thesis
// "Precise Simulations of Multibunches in High Intensity Cyclotrons"
//
// This file is part of OPAL.
//
// OPAL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with OPAL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef IPPL_MPI_DATATYPES_H
#define IPPL_MPI_DATATYPES_H

#include <complex>
#include <cstdint>
#include <mpi.h>

namespace ippl {
    namespace mpi {

        template <typename>
        struct is_ippl_mpi_datatype : std::false_type {};

        template <typename T>
        MPI_Datatype get_mpi_datatype(const T& /*x*/) {
            static_assert(is_ippl_mpi_datatype<T>::value, "type isn't an MPI type");
            return get_mpi_datatype(T());
        }

#define IPPL_MPI_DATATYPE(CppType, MPIType)                         \
    template <>                                                     \
    inline MPI_Datatype get_mpi_datatype<CppType>(const CppType&) { \
        return MPIType;                                             \
    }                                                               \
                                                                    \
    template <>                                                     \
    struct is_ippl_mpi_datatype<CppType> : std::true_type {};

        IPPL_MPI_DATATYPE(char, MPI_CHAR);

        IPPL_MPI_DATATYPE(short, MPI_SHORT);

        IPPL_MPI_DATATYPE(int, MPI_INT);

        IPPL_MPI_DATATYPE(long, MPI_LONG);

        IPPL_MPI_DATATYPE(long long, MPI_LONG_LONG_INT);  // synonym: MPI_LONG_LONG

        IPPL_MPI_DATATYPE(unsigned char, MPI_UNSIGNED_CHAR);

        IPPL_MPI_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT);

        IPPL_MPI_DATATYPE(unsigned int, MPI_UNSIGNED);

        IPPL_MPI_DATATYPE(unsigned long, MPI_UNSIGNED_LONG);

        IPPL_MPI_DATATYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG);

        IPPL_MPI_DATATYPE(float, MPI_FLOAT);

        IPPL_MPI_DATATYPE(double, MPI_DOUBLE);

        IPPL_MPI_DATATYPE(long double, MPI_LONG_DOUBLE);

        IPPL_MPI_DATATYPE(bool, MPI_CXX_BOOL);

        //         IPPL_MPI_DATATYPE(std::int8_t, MPI_INT8_T);

        //         IPPL_MPI_DATATYPE(std::int16_t, MPI_INT16_T);

        //         IPPL_MPI_DATATYPE(std::int32_t, MPI_INT32_T);

        //         IPPL_MPI_DATATYPE(std::int64_t, MPI_INT64_T);

        //         IPPL_MPI_DATATYPE(std::uint8_t, MPI_UINT8_T);

        //         IPPL_MPI_DATATYPE(std::uint16_t, MPI_UINT16_T);

        //         IPPL_MPI_DATATYPE(std::uint32_t, MPI_UINT32_T);

        //         IPPL_MPI_DATATYPE(std::uint64_t, MPI_UINT64_T);

        IPPL_MPI_DATATYPE(std::complex<float>, MPI_C_FLOAT_COMPLEX);

        IPPL_MPI_DATATYPE(std::complex<double>, MPI_C_DOUBLE_COMPLEX);

        IPPL_MPI_DATATYPE(std::complex<long double>, MPI_C_LONG_DOUBLE_COMPLEX);

        IPPL_MPI_DATATYPE(Kokkos::complex<double>, MPI_C_FLOAT_COMPLEX);
        IPPL_MPI_DATATYPE(Kokkos::complex<float>, MPI_C_FLOAT_COMPLEX);
    }  // namespace mpi
}  // namespace ippl

#endif
