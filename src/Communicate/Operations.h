//
// File Operations
//   Definition of MPI operations following the implementation of Boost.MPI.
//
#ifndef IPPL_MPI_OPERATIONS_H
#define IPPL_MPI_OPERATIONS_H

#include <Kokkos_Complex.hpp>
#include <algorithm>
#include <complex>
#include <functional>
#include <map>
#include <mpi.h>
#include <typeindex>
#include <utility>

namespace ippl {
    namespace mpi {

        enum struct binaryOperationKind {
            SUM,
            MIN,
            MAX,
            MULTIPLICATION  // TODO: Add all
        };
        template <typename T>
        struct extractBinaryOperationKind {};
        template <typename T>
        struct extractBinaryOperationKind<std::plus<T>> {
            constexpr static binaryOperationKind value = binaryOperationKind::SUM;
        };
        template <typename T>
        struct extractBinaryOperationKind<std::multiplies<T>> {
            constexpr static binaryOperationKind value = binaryOperationKind::MULTIPLICATION;
        };
        template <typename T>
        struct extractBinaryOperationKind<std::less<T>> {
            constexpr static binaryOperationKind value = binaryOperationKind::MIN;
        };
        template <typename T>
        struct extractBinaryOperationKind<std::greater<T>> {
            constexpr static binaryOperationKind value = binaryOperationKind::MAX;
        };

        template <class>
        struct is_ippl_mpi_type : std::false_type {};
        struct dummy {};
        static std::map<std::pair<std::type_index, std::type_index>, MPI_Op> mpiOperations;
        template <typename CppOpType, typename Datatype_IfNotTrivial>
        struct getMpiOpImpl {
            constexpr MPI_Op operator()() const noexcept {
                static_assert(false, "This optype is not supported");
                return 0; //Dummy return
            }
        };
        /**
         * @brief Helper struct to look up and store MPI_Op types for custom types 
         * 
         */
        template <class Op, typename Type>        
        struct getNontrivialMpiOpImpl /*<Op, Type>*/ {

            /**
             * @brief Get the MPI_Op for this CppOp + Type combo
             * 
             * @return MPI_Op 
             */
            MPI_Op operator()() {
                std::pair<std::type_index, std::type_index> pear{std::type_index(typeid(Op)), std::type_index(typeid(Type))};
                if(mpiOperations.contains(pear)){
                    return mpiOperations.at(pear);
                }
                constexpr binaryOperationKind opKind = extractBinaryOperationKind<Op>::value;
                MPI_Op ret;
                MPI_Op_create(
                    [](void* inputBuffer, void* outputBuffer, int* len, MPI_Datatype*) {
                        Type* input = (Type*)inputBuffer;

                        Type* output = (Type*)outputBuffer;

                        for (int i = 0; i < *len; i++) {
                            if constexpr (opKind == binaryOperationKind::SUM) {
                                output[i] += input[i];
                            }
                            if constexpr (opKind == binaryOperationKind::MIN) {
                                output[i] = min(output[i], input[i]);
                            }
                            if constexpr (opKind == binaryOperationKind::MAX) {
                                output[i] = max(output[i], input[i]);
                            }
                            if constexpr (opKind == binaryOperationKind::MULTIPLICATION) {
                                output[i] *= input[i];
                            }
                        }
                    },
                    1, &ret);
                // static_assert(is_ippl_mpi_type<Op>::value, "type not supported");
                mpiOperations[pear] = ret;
                return ret;
                // return get_mpi_op(op);
            }
        };

#define IPPL_MPI_OP(CppOp, MPIOp)                       \
    template <typename Datatype_IfNotTrivial>           \
    struct getMpiOpImpl<CppOp, Datatype_IfNotTrivial> { \
        constexpr MPI_Op operator()() const noexcept {  \
            return MPIOp;                               \
        }                                               \
    };                                                  \
    template <>                                         \
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

        IPPL_MPI_OP(std::plus<std::complex<float>>, MPI_SUM);
        IPPL_MPI_OP(std::plus<std::complex<double>>, MPI_SUM);
        IPPL_MPI_OP(std::plus<Kokkos::complex<float>>, MPI_SUM);
        IPPL_MPI_OP(std::plus<Kokkos::complex<double>>, MPI_SUM);

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

        template <typename Op, typename Datatype>
        MPI_Op get_mpi_op(Op op) {
            (void)op;
            if constexpr (is_ippl_mpi_type<Op>::value) {
                return getMpiOpImpl<Op, Datatype>{}();
            }
            else {
                return getNontrivialMpiOpImpl<Op, Datatype>{}();
            }
        }
    }  // namespace mpi
}  // namespace ippl

#endif
