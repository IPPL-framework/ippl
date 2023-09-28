//
// Class Ippl
//   Ippl environment.
//
#ifndef IPPL_H
#define IPPL_H

#include <iostream>

#include "Types/IpplTypes.h"

#include "Utility/Inform.h"
#include "Utility/ParallelDispatch.h"

#include "Communicate/Communicate.h"

namespace ippl {

    // the parallel communication object
    // use inlining to avoid multiple definitions
    inline std::unique_ptr<ippl::Communicate> Comm = 0;

    // Inform object to use to print messages to the console (or even to a
    // file if requested)
    // use inlining to avoid multiple definitions
    inline std::unique_ptr<Inform> Info  = 0;
    inline std::unique_ptr<Inform> Warn  = 0;
    inline std::unique_ptr<Inform> Error = 0;

    void initialize(int& argc, char* argv[], MPI_Comm comm = MPI_COMM_WORLD);

    void finalize();

    void fence();

    void abort(const char* msg = nullptr, int errorcode = -1);

    namespace detail {
        bool checkOption(const char* arg, const char* lstr, const char* sstr);

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        T getNumericalOption(const char* arg);
    }  // namespace detail
}  // namespace ippl

// FIMXE remove (only for backwards compatibility)
#include "IpplCore.h"

#endif
