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

                template <typename T, std::contiguous_iterator Iter>
                bool create(const Communicator& comm, Iter first, Iter last);

                template <typename T, std::contiguous_iterator Iter>
                bool attach(const Communicator& comm, Iter first, Iter last);

                template <typename T, std::contiguous_iterator Iter>
                bool detach(Iter first);

                void fence(int asrt = 0);

                template <typename T>
                void put(T* buffer, int count, int dest, unsigned int displ);

                template <typename T>
                void get(T* buffer, int count, int source, unsigned int displ);

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
