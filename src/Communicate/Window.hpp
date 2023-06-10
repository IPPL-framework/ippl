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
namespace ippl {
    namespace mpi {
        namespace rma {

            template <TargetComm Target>
            Window<Target>::~Window() {
                MPI_Win_free(&win_m);
            }

            template <TargetComm Target>
            template <typename T, std::contiguous_iterator Iter>
            bool Window<Target>::create(const Communicator& comm, Iter first, Iter last) {
                static_assert(isActiveTarget<Target>::value,
                              "No active target communication window");

                if (allocated_m) {
                    return false;
                }
                allocated_m = true;

                count_m = std::distance(first, last);
                MPI_Win_create(&first, (MPI_Aint)count_m * sizeof(T), sizeof(T), MPI_INFO_NULL,
                               comm, &win_m);

                return allocated_m;
            }

            template <TargetComm Target>
            void Window<Target>::fence(int asrt) {
                static_assert(isActiveTarget<Target>::value,
                              "No active target communication window");
                MPI_Win_fence(asrt, win_m);
            }

            template <TargetComm Target>
            template <typename T>
            void Window<Target>::put(T* buffer, int count, int dest, unsigned int displ) {
                MPI_Datatype datatype = get_mpi_datatype<T>(*buffer);
                MPI_Put(buffer, count, datatype, dest, displ, count_m, datatype, win_m);
            }

            template <TargetComm Target>
            template <typename T>
            void Window<Target>::get(T* buffer, int count, int source, unsigned int displ) {
                MPI_Datatype datatype = get_mpi_datatype<T>(*buffer);
                MPI_Get(buffer, count, datatype, source, displ, count_m, datatype, win_m);
            }

            template <TargetComm Target>
            template <typename T, std::contiguous_iterator Iter>
            bool Window<Target>::attach(const Communicator& comm, Iter first, Iter last) {
                if (attached_m) {
                    return false;
                }
                attached_m = true;

                if (!allocated_m) {
                    MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &win_m);
                    allocated_m = true;
                }

                count_m = std::distance(first, last);
                MPI_Win_attach(win_m, &first, (MPI_Aint)count_m * sizeof(T));

                return attached_m;
            }

            template <TargetComm Target>
            template <typename T, std::contiguous_iterator Iter>
            bool Window<Target>::detach(Iter first) {
                if (!attached_m) {
                    return false;
                }
                attached_m = false;
                MPI_Win_detach(win_m, &first);
                return true;
            }
        }  // namespace rma
    }      // namespace mpi
}  // namespace ippl
