//
// Class Window
//   Defines an interface to perform one-sided communication.
//   The term RMA stands for remote memory accesss.
//
//
// Copyright (c) 2023, Matthias Frey, University of St Andrews, UK
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
#ifndef IPPL_MPI_WINDOW_H
#define IPPL_MPI_WINDOW_H

#include <iterator>

namespace ippl {
    namespace mpi {
        namespace rma {

            enum TargetComm {
                Active,
                Passive
            };

            template <TargetComm Target>
            struct isActiveTarget : std::false_type {};

            template <>
            struct isActiveTarget<Active> : std::true_type {};

            template <TargetComm Target>
            struct isPassiveTarget : std::false_type {};

            template <>
            struct isPassiveTarget<Passive> : std::true_type {};

            template <TargetComm Target>
            class Window {
            public:
                Window()
                    : count_m(-1)
                    , attached_m(false)
                    , allocated_m(false) {}

                ~Window();

                operator MPI_Win*() noexcept { return &win_m; }

                operator const MPI_Win*() const noexcept { return &win_m; }

                template <std::contiguous_iterator Iter>
                bool create(const Communicator& comm, Iter first, Iter last);

                template <std::contiguous_iterator Iter>
                bool attach(const Communicator& comm, Iter first, Iter last);

                template <std::contiguous_iterator Iter>
                bool detach(Iter first);

                void fence(int asrt = 0);

                template <std::contiguous_iterator Iter>
                void put(Iter first, Iter last, int dest, unsigned int pos,
                         Request* request = nullptr);

                template <typename T>
                void put(const T& value, int dest, unsigned int pos, Request* request = nullptr);

                template <typename T>
                void put(const T* value, int dest, unsigned int pos, Request* request = nullptr);

                template <std::contiguous_iterator Iter>
                void get(Iter first, Iter last, int source, unsigned int pos,
                         Request* request = nullptr);

                template <typename T>
                void get(T& value, int source, unsigned int pos, Request* request = nullptr);

                template <typename T>
                void get(T* value, int source, unsigned int pos, Request* request = nullptr);

                /*
                 * Passive target communication:
                 */
                void flush(int rank);

                void flushall();

                enum LockType : int {
                    Exclusive = MPI_LOCK_EXCLUSIVE,
                    Shared    = MPI_LOCK_SHARED
                };

                void lock(int locktype, int rank, int asrt = 0);

                void lockall(int asrt = 0);

                void unlock(int rank);

                void unlockall();

            private:
                MPI_Win win_m;
                MPI_Aint count_m;
                bool attached_m;
                bool allocated_m;
            };
        }  // namespace rma
    }      // namespace mpi
}  // namespace ippl

#include "Communicate/Window.hpp"

#endif
