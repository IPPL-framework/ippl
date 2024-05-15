//
// File DataTypes
//   Definition of MPI types following the implementation of Boost.MPI.
//
#ifndef IPPL_MPI_DATATYPES_H
#define IPPL_MPI_DATATYPES_H

#include <complex>
#include <cstdint>
#include <mpi.h>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "Utility/IpplException.h"

namespace ippl {
    namespace mpi {
        namespace core {
            static std::unordered_map<std::type_index, MPI_Datatype> type_names = {
                {std::type_index(typeid(std::int8_t)), MPI_INT8_T},
                {std::type_index(typeid(std::int16_t)), MPI_INT16_T},
                {std::type_index(typeid(std::int32_t)), MPI_INT32_T},
                {std::type_index(typeid(std::int64_t)), MPI_INT64_T},

                {std::type_index(typeid(std::uint8_t)), MPI_UINT8_T},
                {std::type_index(typeid(std::uint16_t)), MPI_UINT16_T},
                {std::type_index(typeid(std::uint32_t)), MPI_UINT32_T},
                {std::type_index(typeid(std::uint64_t)), MPI_UINT64_T},

                {std::type_index(typeid(char)), MPI_CHAR},
                {std::type_index(typeid(short)), MPI_SHORT},
                {std::type_index(typeid(int)), MPI_INT},
                {std::type_index(typeid(long)), MPI_LONG},
                {std::type_index(typeid(long long)), MPI_LONG_LONG_INT},  // synonym: MPI_LONG_LONG

                {std::type_index(typeid(unsigned char)), MPI_UNSIGNED_CHAR},
                {std::type_index(typeid(unsigned short)), MPI_UNSIGNED_SHORT},
                {std::type_index(typeid(unsigned int)), MPI_UNSIGNED},
                {std::type_index(typeid(unsigned long)), MPI_UNSIGNED_LONG},
                {std::type_index(typeid(unsigned long long)), MPI_UNSIGNED_LONG_LONG},

                {std::type_index(typeid(float)), MPI_FLOAT},
                {std::type_index(typeid(double)), MPI_DOUBLE},
                {std::type_index(typeid(long double)), MPI_LONG_DOUBLE},

                {std::type_index(typeid(bool)), MPI_CXX_BOOL},

                {std::type_index(typeid(std::complex<float>)), MPI_CXX_FLOAT_COMPLEX},
                {std::type_index(typeid(std::complex<double>)), MPI_CXX_DOUBLE_COMPLEX},
                {std::type_index(typeid(std::complex<long double>)), MPI_CXX_LONG_DOUBLE_COMPLEX},

                {std::type_index(typeid(Kokkos::complex<double>)), MPI_CXX_FLOAT_COMPLEX},
                {std::type_index(typeid(Kokkos::complex<float>)), MPI_CXX_FLOAT_COMPLEX}};
        }
        template <typename T>
        struct vector_dim_type {
            constexpr static unsigned Dim = 0;
        };
        template <typename T, unsigned Dim_>
        struct vector_dim_type<ippl::Vector<T, Dim_>> {
            constexpr static unsigned Dim = Dim_;
        };
        template <typename T>
        MPI_Datatype get_mpi_datatype(const T& /*x*/) {
            MPI_Datatype type = MPI_BYTE;
            if (core::type_names.find(std::type_index(typeid(T))) == core::type_names.end()) {
                if constexpr(vector_dim_type<T>::Dim > 0) {
                    MPI_Datatype tp;
                    MPI_Type_contiguous(vector_dim_type<T>::Dim, get_mpi_datatype<typename T::value_type>(typename T::value_type{}), &tp);
                    MPI_Type_commit(&tp);
                    core::type_names[std::type_index(typeid(T))] = tp;
                }
            }
            try {
                type = core::type_names.at(std::type_index(typeid(T)));
            } catch (...) {
                throw IpplException("ippl::mpi::get_mpi_datatype", "No such type available.");
            }
            return type;
        }
    }  // namespace mpi
}  // namespace ippl

#endif
