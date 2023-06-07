#ifndef IPPL_MPI_STATUS_H
#define IPPL_MPI_STATUS_H

#include <mpi.h>

namespace ippl {
    namespace mpi {

        class Status {
            Status()
                : count_m(-1){};

            int source() const noexcept { return status_m.MPI_SOURCE; }

            int tag() const noexcept { return status_m.MPI_TAG; }

            int error() const noexcept { return status_m.MPI_ERROR; }

            template <typename T>
            std::optional<int> count();

            MPI_Status& operator() { return status_m; }

            const MPI_Status& operator() const { return status_m; }

        private:
            MPI_Status status_m;
            int count_m;
        };

        template <typename T>
        std::optional<int> Status::count() {
            if (count_m != -1) {
                return count_m;
            }

            int count             = MPI_UNDEFINED;
            MPI_Datatype datatype = get_mpi_datatype<T>(T());
            MPI_Get_count(&status_m, datatype, &count);

            if (count == MPI_UNDEFINED) {
                return std::optional<int>();
            }
            count_m = count;
            return count_m;
        }
    }  // namespace mpi
}  // namespace ippl

#endif
