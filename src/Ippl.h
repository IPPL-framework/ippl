//
// Class Ippl
//   Ippl environment.
//
// Copyright (c) 2020, Paul Scherrer Institut, Villigen PSI, Switzerland
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
    inline std::unique_ptr<Inform> Info            = 0;
    inline std::unique_ptr<Inform> Warn            = 0;
    inline std::unique_ptr<Inform> Error           = 0;

    void initialize(int& argc, char* argv[], MPI_Comm comm = MPI_COMM_WORLD);

    void finalize();

    void fence();

    void abort(const std::string&);


    namespace detail {
        bool checkOption(const char* arg, const char* lstr, const char* sstr);

        template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        T getNumericalOption(const char* arg);
    }
}

// FIMXE remove (only for backwards compatibility)
#include "IpplCore.h"

#endif
