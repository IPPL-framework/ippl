#ifndef IPPL_MPI_SERIALIZABLE_H
#define IPPL_MPI_SERIALIZABLE_H

#include <complex>
#include <type_traits>
#include <vector>

namespace ippl {
    namespace mpi {

        template <typename>
        struct is_serializable : std::false_type {};

#define IPPL_MPI_SERIALIZABLE(CppType)                   \
    template <>                                          \
    struct is_serializable<CppType> : std::true_type {}; \
                                                         \
    template <>                                          \
    struct is_serializable<std::vector<CppType> > : std::true_type {};

        IPPL_MPI_SERIALIZABLE(char);

        IPPL_MPI_SERIALIZABLE(short);

        IPPL_MPI_SERIALIZABLE(int);

        IPPL_MPI_SERIALIZABLE(long);

        IPPL_MPI_SERIALIZABLE(long long);

        IPPL_MPI_SERIALIZABLE(unsigned char);

        IPPL_MPI_SERIALIZABLE(unsigned short);

        IPPL_MPI_SERIALIZABLE(unsigned int);

        IPPL_MPI_SERIALIZABLE(unsigned long);

        IPPL_MPI_SERIALIZABLE(unsigned long long);

        IPPL_MPI_SERIALIZABLE(float);

        IPPL_MPI_SERIALIZABLE(double);

        IPPL_MPI_SERIALIZABLE(long double);

        IPPL_MPI_SERIALIZABLE(bool);

        //         IPPL_MPI_SERIALIZABLE(std::int8_t);
        //
        //         IPPL_MPI_SERIALIZABLE(std::int16_t);
        //
        //         IPPL_MPI_SERIALIZABLE(std::int32_t);
        //
        //         IPPL_MPI_SERIALIZABLE(std::int64_t);
        //
        //         IPPL_MPI_SERIALIZABLE(std::uint8_t);
        //
        //         IPPL_MPI_SERIALIZABLE(std::uint16_t);
        //
        //         IPPL_MPI_SERIALIZABLE(std::uint32_t);
        //
        //         IPPL_MPI_SERIALIZABLE(std::uint64_t);

        IPPL_MPI_SERIALIZABLE(std::complex<float>);

        IPPL_MPI_SERIALIZABLE(std::complex<double>);

        IPPL_MPI_SERIALIZABLE(std::complex<long double>);

    }  // namespace mpi
}  // namespace ippl

#endif
