//
// File Operations
//   Definition of MPI operations following the implementation of Boost.MPI.
//
#ifndef IPPL_MPI_OPERATIONS_H
#define IPPL_MPI_OPERATIONS_H

#include <Kokkos_Complex.hpp>
#include <complex>
#include <functional>
#include <mpi.h>

namespace ippl {
    namespace mpi {

        template <class>
        struct is_ippl_mpi_type : std::false_type {};

        template <class Op>
        MPI_Op get_mpi_op(Op op) {
            static_assert(is_ippl_mpi_type<Op>::value, "type not supported");
            return get_mpi_op(op);
        }

#define IPPL_MPI_OP(CppOp, MPIOp)            \
    template <>                              \
    inline MPI_Op get_mpi_op<CppOp>(CppOp) { \
        return MPIOp;                        \
    }                                        \
                                             \
    template <>                              \
    struct is_ippl_mpi_type<CppOp> : std::true_type {};

        /* with C++14 we should be able
         * to simply write
         *
         * IPPL_MPI_OP(std::plus<>, MPI_SUM);
         *
         */

        IPPL_MPI_OP(std::plus<char>, MPI_SUM);
        IPPL_MPI_OP(std::plus<short>, MPI_SUM);
        IPPL_MPI_OP(std::plus<int>, MPI_SUM);
        IPPL_MPI_OP(std::plus<long>, MPI_SUM);
        IPPL_MPI_OP(std::plus<long long>, MPI_SUM);
        IPPL_MPI_OP(std::plus<unsigned char>, MPI_SUM);
        IPPL_MPI_OP(std::plus<unsigned short>, MPI_SUM);
        IPPL_MPI_OP(std::plus<unsigned int>, MPI_SUM);
        IPPL_MPI_OP(std::plus<unsigned long>, MPI_SUM);
        IPPL_MPI_OP(std::plus<unsigned long long>, MPI_SUM);
        IPPL_MPI_OP(std::plus<float>, MPI_SUM);
        IPPL_MPI_OP(std::plus<double>, MPI_SUM);
        IPPL_MPI_OP(std::plus<long double>, MPI_SUM);

        IPPL_MPI_OP(std::plus<std::complex<float> >, MPI_SUM);
        IPPL_MPI_OP(std::plus<std::complex<double> >, MPI_SUM);
        IPPL_MPI_OP(std::plus<Kokkos::complex<float> >, MPI_SUM);
        IPPL_MPI_OP(std::plus<Kokkos::complex<double> >, MPI_SUM);

        IPPL_MPI_OP(std::less<char>, MPI_MIN);
        IPPL_MPI_OP(std::less<short>, MPI_MIN);
        IPPL_MPI_OP(std::less<int>, MPI_MIN);
        IPPL_MPI_OP(std::less<long>, MPI_MIN);
        IPPL_MPI_OP(std::less<long long>, MPI_MIN);
        IPPL_MPI_OP(std::less<unsigned char>, MPI_MIN);
        IPPL_MPI_OP(std::less<unsigned short>, MPI_MIN);
        IPPL_MPI_OP(std::less<unsigned int>, MPI_MIN);
        IPPL_MPI_OP(std::less<unsigned long>, MPI_MIN);
        IPPL_MPI_OP(std::less<unsigned long long>, MPI_MIN);
        IPPL_MPI_OP(std::less<float>, MPI_MIN);
        IPPL_MPI_OP(std::less<double>, MPI_MIN);
        IPPL_MPI_OP(std::less<long double>, MPI_MIN);

        IPPL_MPI_OP(std::greater<char>, MPI_MAX);
        IPPL_MPI_OP(std::greater<short>, MPI_MAX);
        IPPL_MPI_OP(std::greater<int>, MPI_MAX);
        IPPL_MPI_OP(std::greater<long>, MPI_MAX);
        IPPL_MPI_OP(std::greater<long long>, MPI_MAX);
        IPPL_MPI_OP(std::greater<unsigned char>, MPI_MAX);
        IPPL_MPI_OP(std::greater<unsigned short>, MPI_MAX);
        IPPL_MPI_OP(std::greater<unsigned int>, MPI_MAX);
        IPPL_MPI_OP(std::greater<unsigned long>, MPI_MAX);
        IPPL_MPI_OP(std::greater<unsigned long long>, MPI_MAX);
        IPPL_MPI_OP(std::greater<float>, MPI_MAX);
        IPPL_MPI_OP(std::greater<double>, MPI_MAX);
        IPPL_MPI_OP(std::greater<long double>, MPI_MAX);

        IPPL_MPI_OP(std::multiplies<short>, MPI_PROD);
        IPPL_MPI_OP(std::multiplies<int>, MPI_PROD);
        IPPL_MPI_OP(std::multiplies<long>, MPI_PROD);
        IPPL_MPI_OP(std::multiplies<long long>, MPI_PROD);
        IPPL_MPI_OP(std::multiplies<unsigned short>, MPI_PROD);
        IPPL_MPI_OP(std::multiplies<unsigned int>, MPI_PROD);
        IPPL_MPI_OP(std::multiplies<unsigned long>, MPI_PROD);
        IPPL_MPI_OP(std::multiplies<unsigned long long>, MPI_PROD);
        IPPL_MPI_OP(std::multiplies<float>, MPI_PROD);
        IPPL_MPI_OP(std::multiplies<double>, MPI_PROD);
        IPPL_MPI_OP(std::multiplies<long double>, MPI_PROD);

        IPPL_MPI_OP(std::logical_or<bool>, MPI_LOR);
        IPPL_MPI_OP(std::logical_and<bool>, MPI_LAND);
    }  // namespace mpi
}  // namespace ippl

#endif
