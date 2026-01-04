//
// Class Window
//   Defines an interface to perform one-sided communication.
//   The term RMA stands for remote memory accesss.
//
#include "Utility/IpplException.h"

namespace ippl {
    namespace mpi {
        namespace rma {

            template <TargetComm Target>
            Window<Target>::~Window() {
                if (win_m != MPI_WIN_NULL) {
                    MPI_Win_free(&win_m);
                    win_m = MPI_WIN_NULL;
                }
            }

            template <TargetComm Target>
            template <std::contiguous_iterator Iter>
            bool Window<Target>::create(const Communicator& comm, Iter first, Iter last) {
                static_assert(isActiveTarget<Target>::value,
                              "No active target communication window");

                if (allocated_m) {
                    return false;
                }
                allocated_m = true;

                count_m       = std::distance(first, last);
                int dispUnit  = sizeof(typename Iter::value_type);
                MPI_Aint size = (MPI_Aint)count_m * dispUnit;
                MPI_Win_create(&(*first), size, dispUnit, MPI_INFO_NULL, comm, &win_m);

                return allocated_m;
            }

            template <TargetComm Target>
            template <std::contiguous_iterator Iter>
            bool Window<Target>::attach(const Communicator& comm, Iter first, Iter last) {
                if (attached_m) {
                    return false;
                }
                attached_m = true;

                if (!allocated_m) {
                    MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &win_m);
                    allocated_m = true;
                }

                count_m       = std::distance(first, last);
                MPI_Aint size = (MPI_Aint)count_m * sizeof(typename Iter::value_type);
                MPI_Win_attach(win_m, &(*first), size);

                return attached_m;
            }

            template <TargetComm Target>
            template <std::contiguous_iterator Iter>
            bool Window<Target>::detach(Iter first) {
                if (!attached_m) {
                    return false;
                }
                attached_m = false;
                MPI_Win_detach(win_m, &(*first));
                return true;
            }

            template <TargetComm Target>
            void Window<Target>::fence(int asrt) {
                static_assert(isActiveTarget<Target>::value,
                              "No active target communication window");
                MPI_Win_fence(asrt, win_m);
            }

            template <TargetComm Target>
            template <std::contiguous_iterator Iter>
            void Window<Target>::put(Iter first, Iter last, int dest, unsigned int pos,
                                     Request* request) {
                MPI_Datatype datatype = get_mpi_datatype<typename Iter::value_type>(*first);
                auto count            = std::distance(first, last);
                if (count > count_m) {
                    throw IpplException("Window::put", "Count exceeds RMA window size.");
                }
                if (request == nullptr) {
                    MPI_Put(&(*first), count, datatype, dest, (MPI_Aint)pos, count, datatype,
                            win_m);
                } else {
                    MPI_Rput(&(*first), count, datatype, dest, (MPI_Aint)pos, count, datatype,
                             win_m, *request);
                }
            }

            template <TargetComm Target>
            template <typename T>
            void Window<Target>::put(const T* value, int dest, unsigned int pos, Request* request) {
                MPI_Datatype datatype = get_mpi_datatype<T>(*value);
                if (request == nullptr) {
                    MPI_Put(value, 1, datatype, dest, (MPI_Aint)pos, 1, datatype, win_m);
                } else {
                    MPI_Rput(value, 1, datatype, dest, (MPI_Aint)pos, 1, datatype, win_m, *request);
                }
            }

            template <TargetComm Target>
            template <std::contiguous_iterator Iter>
            void Window<Target>::get(Iter first, Iter last, int source, unsigned int pos,
                                     Request* request) {
                MPI_Datatype datatype = get_mpi_datatype<typename Iter::value_type>(*first);
                auto count            = std::distance(first, last);
                if (count > count_m) {
                    throw IpplException("Window::put", "Count exceeds RMA window size.");
                }
                if (request == nullptr) {
                    MPI_Get(&(*first), count, datatype, source, (MPI_Aint)pos, count, datatype,
                            win_m);
                } else {
                    MPI_Rget(&(*first), count, datatype, source, (MPI_Aint)pos, count, datatype,
                             win_m, *request);
                }
            }

            template <TargetComm Target>
            template <typename T>
            void Window<Target>::get(T* value, int source, unsigned int pos, Request* request) {
                MPI_Datatype datatype = get_mpi_datatype<T>(*value);
                if (request == nullptr) {
                    MPI_Get(value, 1, datatype, source, (MPI_Aint)pos, 1, datatype, win_m);
                } else {
                    MPI_Rget(value, 1, datatype, source, (MPI_Aint)pos, 1, datatype, win_m,
                             *request);
                }
            }

            /*
             * Passive target communication:
             */
            template <TargetComm Target>
            void Window<Target>::flush(int rank) {
                static_assert(!isActiveTarget<Target>::value,
                              "No passive target communication window");
                MPI_Win_flush(rank, win_m);
            }

            template <TargetComm Target>
            void Window<Target>::flushall() {
                static_assert(!isActiveTarget<Target>::value,
                              "No passive target communication window");
                MPI_Win_flush_all(win_m);
            }

            template <TargetComm Target>
            void Window<Target>::lock(int locktype, int rank, int asrt) {
                static_assert(!isActiveTarget<Target>::value,
                              "No passive target communication window");
                MPI_Win_lock(locktype, rank, asrt, win_m);
            }

            template <TargetComm Target>
            void Window<Target>::lockall(int asrt) {
                static_assert(!isActiveTarget<Target>::value,
                              "No passive target communication window");
                MPI_Win_lock_all(asrt, win_m);
            }

            template <TargetComm Target>
            void Window<Target>::unlock(int rank) {
                static_assert(!isActiveTarget<Target>::value,
                              "No passive target communication window");
                MPI_Win_unlock(rank, win_m);
            }

            template <TargetComm Target>
            void Window<Target>::unlockall() {
                static_assert(!isActiveTarget<Target>::value,
                              "No passive target communication window");
                MPI_Win_unlock_all(win_m);
            }

        }  // namespace rma
    }      // namespace mpi
}  // namespace ippl
